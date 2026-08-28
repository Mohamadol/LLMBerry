# LLMBerry

A lightweight C++ runtime for LLM inference and decode optimization.

| Doc | Purpose |
|-----|---------|
| [project.md](project.md) | Full 20-checkpoint roadmap |
| [PROGRESS.md](PROGRESS.md) | Checkpoint status and benchmark log |
| [TESTING.md](TESTING.md) | What must pass before each checkpoint is done |

## Quick start

```bash
make setup      # configure + build
make verify     # build + run all tests
make test-tensor # tensor tests only
make run        # llmberry CLI
```

Equivalent scripts: `scripts/setup.sh`, `scripts/verify.sh`, `scripts/benchmark.sh`

```bash
make help       # list all make targets
```

## Checkpoint 01 — Tensor

Implement the methods marked `TODO` in `include/llmberry/tensor.inl`.

**Gate:** `make test-tensor` — 16 tests in `tests/test_tensor.cpp` must pass.
