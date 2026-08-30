"""Plot benchmark JSON produced by python/benchmark.py or `benchmark_matmul --json`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array in {path}")
    return data


def shape_label(row: dict) -> str:
    return f"{row['m']}x{row['n']}\nK={row['k']}"


def plot_file(json_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib is required: pip install matplotlib\n")
        raise

    rows = load_results(json_path)
    if not rows:
        raise ValueError(f"no rows in {json_path}")

    labels = [shape_label(r) for r in rows]
    latency = [r["elapsed_ms"] for r in rows]
    gflops = [r["throughput_gflops"] for r in rows]
    x = range(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, latency)
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Matmul latency")

    axes[1].bar(x, gflops)
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("GFLOP/s")
    axes[1].set_title("Matmul throughput")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot matmul benchmark JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=None,
        help="Input JSON (default: results/benchmarks/matmul/matmul.json)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="PNG output (default: same stem as JSON)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    json_path = (
        args.json_path
        if args.json_path is not None
        else root / "results" / "benchmarks" / "matmul" / "matmul.json"
    )
    if not json_path.is_file():
        sys.stderr.write(f"JSON not found: {json_path}\nRun python/benchmark.py first.\n")
        sys.exit(1)

    png_path = args.out if args.out is not None else json_path.with_suffix(".png")
    plot_file(json_path, png_path)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
