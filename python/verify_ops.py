"""Validate C++ kernels against NumPy / PyTorch references (checkpoint 02).

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


def main() -> None:
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    print("matmul ref:\n", matmul(a, b))
    print("C++ comparison not wired yet")


if __name__ == "__main__":
    main()
