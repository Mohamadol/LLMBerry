"""Run C++ benchmarks and collect JSON results (checkpoint 03)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def benchmark_results_dir(root: Path, name: str) -> Path:
    return root / "results" / "benchmarks" / name


def default_matmul_bin(root: Path) -> Path:
    return root / "build" / "benchmarks" / "benchmark_matmul"


def run_matmul(bin_path: Path, large: bool) -> list[dict]:
    cmd = [str(bin_path), "--json"]
    if large:
        cmd.append("--large")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LLMBerry matmul benchmark JSON.")
    parser.add_argument(
        "--large",
        action="store_true",
        help="Include the slow {256, 4096, 4096} case",
    )
    parser.add_argument(
        "--bin",
        type=Path,
        default=None,
        help="Path to benchmark_matmul (default: build/benchmarks/benchmark_matmul)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path (default: results/benchmarks/matmul/matmul.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a PNG via plot_results.py",
    )
    args = parser.parse_args()

    root = repo_root()
    bin_path = args.bin if args.bin is not None else default_matmul_bin(root)
    if not bin_path.is_file():
        sys.stderr.write(f"benchmark binary not found: {bin_path}\nRun `make build` first.\n")
        sys.exit(1)

    out_path = (
        args.out
        if args.out is not None
        else benchmark_results_dir(root, "matmul") / "matmul.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_matmul(bin_path, large=args.large)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out_path} ({len(results)} rows)")

    if args.plot:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_results import plot_file

        png = out_path.with_suffix(".png")
        plot_file(out_path, png)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
