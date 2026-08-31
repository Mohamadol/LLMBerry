"""Validate C++ kernels against NumPy / PyTorch references (checkpoints 02, 04, 05).

Fill in a comparison path once kernels are implemented (golden files from
C++ tests, or a small binding). Until then these functions are the numeric
reference for the naive CPU ops.
"""

from __future__ import annotations

import math

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


def main() -> None:
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    print("matmul ref:\n", matmul(a, b))

    x = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
    w = np.ones(4, dtype=np.float32)
    print("rmsnorm ref:\n", rmsnorm(x, w))

    q = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
    print("rope freqs dim=4:\n", rope_freqs(4))
    print("rope ref:\n", rope(q))
    print("C++ comparison not wired yet")


if __name__ == "__main__":
    main()
