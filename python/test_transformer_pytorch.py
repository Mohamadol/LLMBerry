"""Random C++ Transformer vs PyTorch LlamaDecoderBlock.

C++ `dump_ops` draws tiny and Llama-like (x, weights) cases. This script rebuilds
the same block as a PyTorch `nn.Module` and checks allclose.

Needs PyTorch in a venv (Homebrew Python cannot pip-install it system-wide):

  make python-deps
  .venv/bin/python python/test_transformer_pytorch.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_ops import (  # noqa: E402
    allclose,
    cpp_output,
    default_dump_bin,
    inputs_from_json,
    load_cpp_dump,
)

EXPECTED_TINY = 5
EXPECTED_REAL = 6


def pytorch_block(op: dict) -> np.ndarray:
    from llama_block_torch import llama_decoder_block

    inp = inputs_from_json(op)
    attrs = op.get("attrs", {})
    return llama_decoder_block(
        inp["x"],
        inp["attn_norm"],
        inp["w_q"],
        inp["w_k"],
        inp["w_v"],
        inp["w_o"],
        inp["ffn_norm"],
        inp["w_gate"],
        inp["w_up"],
        inp["w_down"],
        int(attrs["n_heads"]),
        int(attrs["n_kv_heads"]),
        int(attrs.get("position_offset", 0)),
        float(attrs.get("theta", 10000.0)),
        float(attrs.get("eps", 1e-6)),
    )


def _group(ops: list[dict], prefix: str) -> list[dict]:
    grouped = [op for op in ops if op["name"].startswith(prefix)]
    grouped.sort(key=lambda op: op["name"])
    return grouped


def _run_group(ops: list[dict], atol: float, rtol: float) -> int:
    failed = 0
    for i, op in enumerate(ops):
        cpp = cpp_output(op)
        pt = pytorch_block(op)
        ok, msg = allclose(cpp, pt, atol, rtol)
        status = "PASS" if ok else "FAIL"
        extra = f"  {msg}" if msg else ""
        attrs = op.get("attrs", {})
        n_heads = int(attrs["n_heads"])
        n_kv = int(attrs["n_kv_heads"])
        pos = int(attrs.get("position_offset", 0))
        shape = tuple(np.asarray(cpp).shape)
        hidden = int(shape[-1])
        head_dim = hidden // n_heads
        print(
            f"  [{status}] {op['name']:22}  x={shape}  "
            f"H={hidden} heads={n_heads} kv={n_kv} d={head_dim} pos={pos}{extra}"
        )
        if not ok:
            failed += 1
    return failed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Random Transformer cases vs PyTorch.")
    parser.add_argument("--bin", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Exit 0 if PyTorch is missing (used by CTest).",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        msg = (
            "PyTorch is not installed (Homebrew Python cannot pip-install it system-wide).\n"
            "  make python-deps\n"
            "  .venv/bin/python python/test_transformer_pytorch.py"
        )
        if args.allow_skip:
            print(f"SKIP: {msg}")
            return 0
        print(msg, file=sys.stderr)
        return 1
    bin_path = args.bin if args.bin is not None else default_dump_bin(root)

    dump = load_cpp_dump(bin_path)
    tiny = _group(dump["ops"], "transformer_random_")
    real = _group(dump["ops"], "transformer_real_")
    if len(tiny) != EXPECTED_TINY:
        print(f"FAIL: expected {EXPECTED_TINY} transformer_random_* ops, got {len(tiny)}")
        return 1
    if len(real) != EXPECTED_REAL:
        print(f"FAIL: expected {EXPECTED_REAL} transformer_real_* ops, got {len(real)}")
        return 1

    print(f"dump_ops: {bin_path}")
    print(f"Tiny shapes ({len(tiny)} cases) vs PyTorch LlamaDecoderBlock")
    failed = _run_group(tiny, args.atol, args.rtol)
    print(f"Realistic Llama-like shapes ({len(real)} cases) vs PyTorch LlamaDecoderBlock")
    failed += _run_group(real, args.atol, args.rtol)

    total = len(tiny) + len(real)
    if failed:
        print(f"\n{failed}/{total} case(s) failed")
        return 1
    print(f"\nAll {total} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
