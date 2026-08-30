# LLMBerry Progress

Track checkpoint status, test results, and benchmark notes here. Update this file when a checkpoint is done or when you record a new benchmark.

Full roadmap: [project.md](project.md)

---

## 01 — Tensor Abstraction

**Status:** Complete

**Completed:**
- Owning constructor (shape, row-major strides, `shared_ptr` storage)
- View constructor (external buffer or shared storage)
- `size()`, `nbytes()`, `data()`
- `zeros`, `ones`
- Flat indexing (`operator[]`) and multi-dim indexing (`at`)
- `offset_from_indices`, `validate_shape`, `compute_row_major_strides`, `validate_indices`
- `view()` reshape sharing storage

**Tests:**
- `test_tensor`: 17/17 passing (`make test-tensor`)
- Other test binaries: skipped (not implemented yet)

**Notes:**
- Owning tensors use `std::shared_ptr<std::vector<T>>` for RAII.
- External views do not allocate; caller owns the buffer (`storage` is null).
- Views from owning tensors pass `storage_` into the view constructor so the buffer stays alive.

---

## 02 — CPU Kernels

**Status:** Complete

**Completed:**
- Naive packed row-major kernels: `matmul`, `gemv`, `add`, `mul`, `dot`, `reduce_sum`, `reduce_mean`, `silu`
- Extra: exact (erf) `gelu`
- Shape / rank checks via `ENSURE` (`include/llmberry/ensure.h`)
- Allocating wrappers for `add`, `mul`, `silu`, `gelu`
- C++ tests vs independent references (`at()` GEMM, `exp`/`erf` activations)

**Tests:**
- `test_matmul`: 32/32 passing (`make test-matmul`)
- Later binaries (`test_rmsnorm`, `test_rope`, `test_attention`): still skipped

**Notes:**
- Kernels write into caller-owned outputs (no realloc of `out` / `C`).
- GEMM/GEMV assume inner stride 1.
- `python/verify_ops.py` has NumPy helpers; there is no C++ binding yet — gtests are the correctness gate.

---

## 03 — Benchmarking Infrastructure

**Status:** Not started

**Notes:**
- Scaffold exists in `benchmarks/` and `python/benchmark.py`

---

## Checkpoints 04–20

**Status:** Not started

See [project.md](project.md) for RMSNorm, RoPE, attention, transformer block, weight loading, generation, KV cache, profiling, optimization, quantization, batching, and paged KV cache.

---

## Benchmark log

Record numbers here as you measure. Example format:

```text
## 02 — matmul (naive)
Date: YYYY-MM-DD
Matrix: 512 × 512
Threads: 1
Latency: TBD ms
```
