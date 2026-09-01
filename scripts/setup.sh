#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"

cmake -B "${ROOT}/build" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${ROOT}/build" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "Build complete. Run: make test  or  make verify"
