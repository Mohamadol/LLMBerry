#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"

if [[ ! -x "${BUILD}/benchmarks/benchmark_matmul" ]]; then
  "${ROOT}/scripts/setup.sh"
fi

echo "=== benchmark_matmul ==="
"${BUILD}/benchmarks/benchmark_matmul"
echo ""
echo "=== benchmark_attention ==="
"${BUILD}/benchmarks/benchmark_attention"
echo ""
echo "=== benchmark_decode ==="
"${BUILD}/benchmarks/benchmark_decode"
