"""PyTorch Llama decoder layer. Weights are loaded from LLMBerry's compute-native layout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_sincos(
    seq: int,
    dim: int,
    position_offset: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (i / dim))
    positions = torch.arange(
        position_offset, position_offset + seq, device=device, dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq, n_heads, dim]; cos/sin: [seq, dim]
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    return x * cos + _rotate_half(x) * sin


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


class LlamaDecoderBlock(nn.Module):
    """Llama decoder layer using nn.Linear (Hugging Face [out, in] weight layout)."""

    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate: int,
        eps: float = 1e-6,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden // n_heads
        self.eps = eps
        self.theta = theta
        self.input_layernorm_weight = nn.Parameter(torch.empty(hidden))
        self.q_proj = nn.Linear(hidden, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden, bias=False)
        self.post_attention_layernorm_weight = nn.Parameter(torch.empty(hidden))
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def load_cpp_weights(
        self,
        attn_norm: np.ndarray,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        w_o: np.ndarray,
        ffn_norm: np.ndarray,
        w_gate: np.ndarray,
        w_up: np.ndarray,
        w_down: np.ndarray,
    ) -> None:
        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.float32).copy())

        # C++ stores [in, out]; nn.Linear stores [out, in].
        self.input_layernorm_weight.data.copy_(t(attn_norm))
        self.q_proj.weight.data.copy_(t(w_q).T.contiguous())
        self.k_proj.weight.data.copy_(t(w_k).T.contiguous())
        self.v_proj.weight.data.copy_(t(w_v).T.contiguous())
        self.o_proj.weight.data.copy_(t(w_o).T.contiguous())
        self.post_attention_layernorm_weight.data.copy_(t(ffn_norm))
        self.gate_proj.weight.data.copy_(t(w_gate).T.contiguous())
        self.up_proj.weight.data.copy_(t(w_up).T.contiguous())
        self.down_proj.weight.data.copy_(t(w_down).T.contiguous())

    def forward(self, hidden_states: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.input_layernorm_weight, self.eps)

        bsz, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq, self.n_kv_heads, self.head_dim)

        cos, sin = _rope_sincos(seq, self.head_dim, position_offset, self.theta, hidden_states.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.n_heads != self.n_kv_heads:
            group = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(bsz, seq, self.hidden)
        hidden_states = residual + self.o_proj(attn)

        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.post_attention_layernorm_weight, self.eps)
        hidden_states = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return residual + hidden_states


def llama_decoder_block(
    x: np.ndarray,
    attn_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    ffn_norm: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    position_offset: int = 0,
    theta: float = 10000.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Run `LlamaDecoderBlock` on NumPy inputs. Returns a NumPy array matching `x`."""
    x = np.asarray(x, dtype=np.float32)
    hidden = int(x.shape[-1])
    intermediate = int(np.asarray(w_gate).shape[1])
    block = LlamaDecoderBlock(hidden, n_heads, n_kv_heads, intermediate, eps, theta)
    block.load_cpp_weights(attn_norm, w_q, w_k, w_v, w_o, ffn_norm, w_gate, w_up, w_down)
    block.eval()

    t = torch.from_numpy(x.copy())
    squeeze_batch = False
    squeeze_seq = False
    if t.ndim == 1:
        t = t.view(1, 1, -1)
        squeeze_batch = True
        squeeze_seq = True
    elif t.ndim == 2:
        t = t.unsqueeze(0)
        squeeze_batch = True
    elif t.ndim != 3:
        raise ValueError(f"expected x rank 1–3, got {tuple(t.shape)}")

    with torch.no_grad():
        y = block(t, position_offset)
    if squeeze_seq:
        y = y[0, 0]
    elif squeeze_batch:
        y = y[0]
    return y.contiguous().numpy()
