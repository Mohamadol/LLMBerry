# LLMBerry

A lightweight C++ runtime for LLM inference and decode optimization.

| Doc | Purpose |
|-----|---------|
| [project.md](project.md) | Full 20-checkpoint roadmap |
| [PROGRESS.md](PROGRESS.md) | Checkpoint status, **Verify** commands, benchmark log |
| [TESTING.md](TESTING.md) | What must pass before each checkpoint is done |

## Quick start

```bash
make setup       # configure + build
make verify      # build + run all tests
make test-tensor  # checkpoint 01
make test-matmul  # checkpoint 02
make benchmark    # checkpoint 03 (matmul default + stubs)
make test-rmsnorm # checkpoint 04
make test-rope    # checkpoint 05
make test-attention # checkpoint 06–07
make test-mlp       # checkpoint 08
make test-transformer  # checkpoint 09
make verify-ops     # C++ vs NumPy (dump_ops)
make run         # llmberry CLI
```

Per-checkpoint commands (including `benchmark_matmul --large`) are at the end of each section in [PROGRESS.md](PROGRESS.md).

Equivalent scripts: `scripts/setup.sh`, `scripts/verify.sh`, `scripts/benchmark.sh`

```bash
make help       # list all make targets
```

## Checkpoint 01 — Tensor

Implement the methods marked `TODO` in `include/llmberry/tensor.inl`.

**Gate:** `make test-tensor` — 19 tests in `tests/test_tensor.cpp` must pass.
