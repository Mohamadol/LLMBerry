#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${ROOT}/build" ]]; then
  "${ROOT}/scripts/setup.sh"
else
  cmake --build "${ROOT}/build" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

ctest --test-dir "${ROOT}/build" --output-on-failure
