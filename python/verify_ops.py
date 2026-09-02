"""Validate C++ kernels against NumPy (and PyTorch if installed).

Runs `build/tests/dump_ops`, which prints JSON of C++ inputs + outputs.
This script recomputes each op with NumPy from those same inputs and checks
allclose. Optional: the same check with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)


def gemv(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    return a @ x


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def reduce_sum(x: np.ndarray) -> float:
    return float(np.sum(x))


def reduce_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU (PyTorch approximate='none')."""
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Llama RMSNorm: y = x / sqrt(mean(x^2, last_dim) + eps) * weight.

    Matches Hugging Face LlamaRMSNorm (variance in float32, rsqrt, eps=1e-6).
    `x` is [..., D], `weight` is [D].
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(variance + np.float32(eps))) * weight


def rope_freqs(dim: int, theta: float = 10000.0) -> np.ndarray:
    """Llama inv_freq: 1 / theta^(2i / dim), shape [dim/2]."""
    i = np.arange(0, dim, 2, dtype=np.float32)
    return (1.0 / (np.float32(theta) ** (i / np.float32(dim)))).astype(np.float32)


def rope_sincos(
    seq_len: int,
    dim: int,
    position_offset: int = 0,
    theta: float = 10000.0,
    positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """HF Llama cos/sin tables: concat(freqs, freqs), shape [n_pos, dim]."""
    inv_freq = rope_freqs(dim, theta)
    if positions is None:
        positions = np.arange(position_offset, position_offset + seq_len, dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # [n_pos, dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Llama / GPT-NeoX pairing: concat(-x[..., d/2:], x[..., :d/2])."""
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rope(
    x: np.ndarray,
    position_offset: int = 0,
    theta: float = 10000.0,
    cos: np.ndarray | None = None,
    sin: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Llama RoPE. `x` is [seq, ..., head_dim]; dim 0 is sequence."""
    x = np.asarray(x, dtype=np.float32)
    seq, dim = x.shape[0], x.shape[-1]
    if cos is None or sin is None:
        cos, sin = rope_sincos(seq, dim, position_offset, theta)
    bcast = (seq,) + (1,) * (x.ndim - 2) + (dim,)
    return x * cos.reshape(bcast) + rotate_half(x) * sin.reshape(bcast)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis. Matches C++ `softmax`."""
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    """Scaled dot-product attention. Matches PyTorch `F.scaled_dot_product_attention`.

    q: [..., seq_q, d], k: [..., seq_k, d], v: [..., seq_k, d_v]
    GQA: if n_q > n_kv, K/V heads are repeat-interleaved along the head axis.
    Causal mask is bottom-right aligned (decode seq_q=1 sees every key).
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if q.ndim >= 3 and q.shape[-3] != k.shape[-3]:
        n_q, n_kv = q.shape[-3], k.shape[-3]
        k = np.repeat(k, n_q // n_kv, axis=-3)
        v = np.repeat(v, n_q // v.shape[-3], axis=-3)
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(np.float32(d))
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(scale)
    if causal:
        seq_q, seq_k = q.shape[-2], k.shape[-2]
        q_idx = np.arange(seq_q)[:, None]
        k_idx = np.arange(seq_k)[None, :]
        mask = k_idx > (q_idx + seq_k - seq_q)
        scores = np.where(mask, np.float32(-np.inf), scores)
    return np.matmul(softmax(scores), v)


def mlp(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """Llama SwiGLU MLP. Matches C++ `mlp` (compute-native weight layout).

    x:      [..., hidden]
    w_gate: [hidden, intermediate]
    w_up:   [hidden, intermediate]
    w_down: [intermediate, hidden]

    Hugging Face Linear stores [out, in]; pass W.T:
      mlp(x, gate.weight.T, up.weight.T, down.weight.T)
    """
    x = np.asarray(x, dtype=np.float32)
    w_gate = np.asarray(w_gate, dtype=np.float32)
    w_up = np.asarray(w_up, dtype=np.float32)
    w_down = np.asarray(w_down, dtype=np.float32)
    hidden = silu(x @ w_gate) * (x @ w_up)
    return hidden @ w_down


def tensor_from_json(obj: dict) -> np.ndarray:
    return np.asarray(obj["data"], dtype=np.float32).reshape(obj["shape"])


def inputs_from_json(op: dict) -> dict[str, np.ndarray]:
    return {name: tensor_from_json(t) for name, t in op.get("inputs", {}).items()}


def numpy_ref(op: dict):
    name = op["name"]
    inp = inputs_from_json(op)
    attrs = op.get("attrs", {})
    if name == "matmul":
        return matmul(inp["a"], inp["b"])
    if name == "gemv":
        return gemv(inp["a"], inp["x"])
    if name == "add":
        return add(inp["a"], inp["b"])
    if name == "mul":
        return mul(inp["a"], inp["b"])
    if name == "dot":
        return dot(inp["a"], inp["b"])
    if name == "reduce_sum":
        return reduce_sum(inp["x"])
    if name == "reduce_mean":
        return reduce_mean(inp["x"])
    if name == "silu":
        return silu(inp["x"])
    if name == "gelu":
        return gelu(inp["x"])
    if name == "rmsnorm":
        return rmsnorm(inp["x"], inp["weight"], float(attrs.get("eps", 1e-6)))
    if name == "rope_freqs":
        return rope_freqs(int(attrs["dim"]), float(attrs["theta"]))
    if name == "rope_sincos":
        return rope_sincos(
            int(attrs["seq_len"]),
            int(attrs["dim"]),
            int(attrs.get("position_offset", 0)),
            float(attrs.get("theta", 10000.0)),
        )
    if name == "rope":
        return rope(
            inp["x"],
            int(attrs.get("position_offset", 0)),
            float(attrs.get("theta", 10000.0)),
        )
    if name == "softmax":
        return softmax(inp["x"])
    if name in ("attention", "attention_gqa"):
        return attention(
            inp["q"],
            inp["k"],
            inp["v"],
            causal=bool(attrs.get("causal", True)),
        )
    if name in ("mlp", "mlp_batched"):
        return mlp(inp["x"], inp["w_gate"], inp["w_up"], inp["w_down"])
    raise KeyError(f"unknown op {name}")


def torch_ref(op: dict):
    import torch
    import torch.nn.functional as F

    name = op["name"]
    inp = {k: torch.from_numpy(v.copy()) for k, v in inputs_from_json(op).items()}
    attrs = op.get("attrs", {})

    if name == "matmul":
        return (inp["a"] @ inp["b"]).numpy()
    if name == "gemv":
        return (inp["a"] @ inp["x"]).numpy()
    if name == "add":
        return (inp["a"] + inp["b"]).numpy()
    if name == "mul":
        return (inp["a"] * inp["b"]).numpy()
    if name == "dot":
        return float(torch.dot(inp["a"].reshape(-1), inp["b"].reshape(-1)))
    if name == "reduce_sum":
        return float(inp["x"].sum())
    if name == "reduce_mean":
        return float(inp["x"].mean())
    if name == "silu":
        return F.silu(inp["x"]).numpy()
    if name == "gelu":
        return F.gelu(inp["x"], approximate="none").numpy()
    if name == "rmsnorm":
        x, w = inp["x"], inp["weight"]
        eps = float(attrs.get("eps", 1e-6))
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), w, eps).numpy()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + eps) * w).numpy()
    if name == "softmax":
        return F.softmax(inp["x"], dim=-1).numpy()
    if name in ("attention", "attention_gqa"):
        q, k, v = inp["q"], inp["k"], inp["v"]
        if q.ndim >= 3 and q.shape[-3] != k.shape[-3]:
            n_q, n_kv = q.shape[-3], k.shape[-3]
            k = k.repeat_interleave(n_q // n_kv, dim=-3)
            v = v.repeat_interleave(n_q // n_kv, dim=-3)
        squeeze_batch = q.ndim == 2
        if squeeze_batch:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=bool(attrs.get("causal", True)))
        if squeeze_batch:
            y = y.squeeze(0)
        return y.numpy()
    if name in ("mlp", "mlp_batched"):
        x, wg, wu, wd = inp["x"], inp["w_gate"], inp["w_up"], inp["w_down"]
        h = F.silu(x @ wg) * (x @ wu)
        return (h @ wd).numpy()
    if name in ("rope", "rope_freqs", "rope_sincos"):
        return None  # no torch.nn RoPE; NumPy matches HF rotate_half
    raise KeyError(f"unknown op {name}")


def cpp_output(op: dict):
    if "value" in op:
        return float(op["value"])
    if "outputs" in op:
        return {k: tensor_from_json(v) for k, v in op["outputs"].items()}
    return tensor_from_json(op["output"])


def allclose(got, want, atol: float, rtol: float) -> tuple[bool, str]:
    if isinstance(want, tuple):
        if not isinstance(got, dict) or got.keys() != {"cos", "sin"}:
            return False, f"expected cos/sin dict, got {type(got)}"
        ok_c = np.allclose(got["cos"], want[0], atol=atol, rtol=rtol)
        ok_s = np.allclose(got["sin"], want[1], atol=atol, rtol=rtol)
        if ok_c and ok_s:
            return True, ""
        err = max(
            float(np.max(np.abs(got["cos"] - want[0]))),
            float(np.max(np.abs(got["sin"] - want[1]))),
        )
        return False, f"max abs err {err:.3e}"
    if isinstance(want, float):
        if not math.isclose(float(got), want, abs_tol=atol, rel_tol=rtol):
            return False, f"cpp={got!r} numpy={want!r}"
        return True, ""
    got_a = np.asarray(got, dtype=np.float32)
    want_a = np.asarray(want, dtype=np.float32)
    if got_a.shape != want_a.shape:
        return False, f"shape cpp {got_a.shape} vs numpy {want_a.shape}"
    if np.allclose(got_a, want_a, atol=atol, rtol=rtol, equal_nan=True):
        return True, ""
    err = float(np.max(np.abs(got_a - want_a)))
    return False, f"max abs err {err:.3e}"


def default_dump_bin(root: Path) -> Path:
    return root / "build" / "tests" / "dump_ops"


def load_cpp_dump(bin_path: Path) -> dict:
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"dump_ops not found at {bin_path}. Build it with: cmake --build build --target dump_ops"
        )
    proc = subprocess.run([str(bin_path)], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def compare_ops(dump: dict, atol: float, rtol: float, use_torch: bool) -> int:
    failed = 0
    torch_mod = None
    if use_torch:
        try:
            import torch  # noqa: F401

            torch_mod = True
        except ImportError:
            print("PyTorch not installed — skipping C++ vs PyTorch")

    for op in dump["ops"]:
        name = op["name"]
        cpp = cpp_output(op)
        np_want = numpy_ref(op)
        ok, msg = allclose(cpp, np_want, atol, rtol)
        status = "PASS" if ok else "FAIL"
        extra = f"  {msg}" if msg else ""
        print(f"  [{status}] {name:16}  C++ vs NumPy{extra}")
        if not ok:
            failed += 1
            if not isinstance(np_want, (float, tuple)):
                print("    numpy:\n", np_want)
                print("    cpp:\n", cpp)

        if torch_mod:
            try:
                th_want = torch_ref(op)
            except KeyError:
                th_want = None
            if th_want is None:
                print(f"  [SKIP] {name:16}  PyTorch (no ref)")
                continue
            ok_t, msg_t = allclose(cpp, th_want, atol, rtol)
            st = "PASS" if ok_t else "FAIL"
            extra_t = f"  {msg_t}" if msg_t else ""
            print(f"  [{st}] {name:16}  C++ vs PyTorch{extra_t}")
            if not ok_t:
                failed += 1

    return failed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare dump_ops C++ JSON against NumPy / PyTorch.")
    parser.add_argument(
        "--bin",
        type=Path,
        default=None,
        help="Path to dump_ops (default: build/tests/dump_ops)",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--no-torch", action="store_true", help="Do not try PyTorch")
    args = parser.parse_args()
    bin_path = args.bin if args.bin is not None else default_dump_bin(root)

    dump = load_cpp_dump(bin_path)
    print(f"dump_ops: {bin_path}  ({len(dump['ops'])} ops)")
    failed = compare_ops(dump, args.atol, args.rtol, use_torch=not args.no_torch)
    if failed:
        print(f"\n{failed} comparison(s) failed")
        return 1
    print("\nAll comparisons passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
