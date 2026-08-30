# LLMBerry Testing

What must pass before you consider a checkpoint done. Run everything with:

```bash
make verify
```

---

## Quick commands

| Command | What it runs |
|---------|----------------|
| `make test` | All CTest targets |
| `make test-tensor` | `test_tensor` only |
| `make test-tensor-construction` | Construction / metadata tests |
| `make test-tensor-indexing` | Flat and multi-dim indexing |
| `make test-tensor-views` | View and reshape tests |
| `make test-matmul` | Checkpoint 02 kernel tests (matmul, gemv, elemwise, reduce, SiLU, GELU) |
| `make verify` | Build (if needed) + full test suite |

---

## Checkpoint 01 — Tensor

**Gate:** all tests in `tests/test_tensor.cpp` pass (16 tests).

| Group | Filter | Count | Covers |
|-------|--------|-------|--------|
| Construction | `TensorConstruction.*` | 3 | shape, strides, size, nbytes, dtype, zero-dim rejection |
| Dtype | `TensorDtype.*` | 1 | `dtype_of` / `Tensor::dtype()` |
| Factories | `TensorFactories.*` | 2 | `zeros`, `ones` |
| Flat indexing | `TensorIndexing.Flat*` | 2 | `operator[]`, bounds |
| Multi-dim | `TensorIndexing.At*` | 4 | `at()`, rank / bounds errors |
| Data pointer | `TensorData.*` | 1 | `data()` row-major layout |
| Views | `TensorView.*` | 3 | reshape view, invalid reshape, external buffer |
| Layout | `TensorLayout.*` | 1 | row-major linearization |

**Manual smoke test:**

```bash
./build/llmberry
```

---

## Checkpoint 02 — CPU Kernels

**Gate:** all tests in `tests/test_matmul.cpp` pass (`make test-matmul`).

| Group | Filter | Covers |
|-------|--------|--------|
| Matmul | `Matmul.*` | GEMM, shape errors |
| Gemv | `Gemv.*` | matrix-vector |
| Elemwise | `Elemwise.*` | add, mul |
| Dot / reduce | `Dot.*`, `Reduce.*` | dot, sum, mean |
| SiLU | `Silu.*` | activation |
| GELU | `Gelu.*` | exact / erf GELU |

**Also:** `python/verify_ops.py` matches NumPy / PyTorch within tolerance.

---

## Checkpoint 03 — Benchmarking

**Gate:** benchmark binaries build and produce repeatable timings.

```bash
make build
./build/benchmarks/benchmark_matmul
./build/benchmarks/benchmark_matmul --large
./build/benchmarks/benchmark_matmul --json
python3 python/benchmark.py
python3 python/plot_results.py
make benchmark
```

`--large` adds `{256, 4096, 4096}`. `--json` prints a JSON array to stdout. `python/benchmark.py --plot` writes `results/benchmarks/matmul/`. `make benchmark` uses the default matmul set (no `--large`).

---

## Checkpoints 04–07

| Checkpoint | Test binary | Gate |
|------------|-------------|------|
| 04 RMSNorm | `test_rmsnorm` | matches PyTorch reference |
| 05 RoPE | `test_rope` | matches Python reference |
| 06 Attention | `test_attention` | matches PyTorch |
| 07 GQA | `test_attention` (extended) | GQA head layout correct |

---

## Later checkpoints

Add gates here as you add tests:

- **10** — weights load without error; tensor shapes match config
- **11** — `python/verify_logits.py` matches Hugging Face logits
- **12** — CLI generates text from a real prompt
- **13** — KV cache decode faster than no-cache baseline
- **14** — separate prefill / decode paths benchmarked
- **18** — quantized weights run and stay within accuracy tolerance
- **19–20** — multi-request tests + paged KV utilization metrics

---

## CI expectation (future)

When CI is added, it should run:

```bash
make setup
make verify
```
