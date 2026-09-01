# LLMBerry Progress

Track checkpoint status, test results, and benchmark notes here. Update this file when a checkpoint is done or when you record a new benchmark.

Full roadmap: [project.md](project.md). Gates and test filters: [TESTING.md](TESTING.md).

Each checkpoint ends with **Verify** — the commands to run for that case.

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
- `row(i)` — 1-D view of last-dimension vector `i` (contiguous last dim)

**Tests:**
- `test_tensor`: 19/19 passing (`make test-tensor`)

**Notes:**
- Owning tensors use `std::shared_ptr<std::vector<T>>` for RAII.
- External views do not allocate; caller owns the buffer (`storage` is null).
- Views from owning tensors pass `storage_` into the view constructor so the buffer stays alive.

**Verify:**

```bash
make test-tensor
make test-tensor-construction
make test-tensor-indexing
make test-tensor-views
./build/llmberry
```

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

**Notes:**
- Kernels write into caller-owned outputs (no realloc of `out` / `C`).
- GEMV assumes inner stride 1. `matmul` uses both strides (supports a transposed `B`).
- `python/verify_ops.py` has NumPy helpers; there is no C++ binding yet — gtests are the correctness gate.

**Verify:**

```bash
make test-matmul
python3 python/verify_ops.py
```

---

## 03 — Benchmarking Infrastructure

**Status:** Not started

**Notes:**
- Scaffold exists in `benchmarks/` and `python/benchmark.py`
- Matmul bench times naive GEMM; `--large` adds `{256, 4096, 4096}` (slow, DRAM-bound)

**Verify:**

```bash
make build

# human-readable timings (stdout)
./build/benchmarks/benchmark_matmul
./build/benchmarks/benchmark_matmul --large
./build/benchmarks/benchmark_matmul --json

# JSON + plot under results/benchmarks/matmul/
pip install matplotlib
python3 python/benchmark.py --plot
python3 python/plot_results.py
python3 python/benchmark.py --large --plot

# all C++ benches (matmul default set, no --large / --json; attention/decode stubs)
make benchmark
```

Writes `results/benchmarks/matmul/matmul.json` and `results/benchmarks/matmul/matmul.png`.

---

## 04 — RMSNorm

**Status:** Complete

**Completed:**
- Llama RMSNorm over the last dim: `y = x / sqrt(mean(x^2) + eps) ⊙ weight`
- In-place `rmsnorm(x, weight, out, eps=1e-6)` plus allocating wrapper
- Shape / rank checks; last-dim row views via `Tensor::row`
- Numerical stability: `eps` added before `sqrt`; zero and tiny inputs stay finite
- C++ tests vs independent last-dim reference; NumPy/HF reference in `python/verify_ops.py`

**Tests:**
- `test_rmsnorm`: 13/13 passing (`make test-rmsnorm`)

**Notes:**
- Default `eps` matches Hugging Face LlamaRMSNorm (`1e-6`).
- Each leading-dim vector is normalized independently, then scaled by 1-D `weight`.
- Scale/write uses `std::transform` over contiguous row pointers.

**Verify:**

```bash
make test-rmsnorm
python3 python/verify_ops.py
```

---

## 05 — Rotary Positional Embeddings

**Status:** Scaffold ready — implement frequencies, sincos tables, and apply

**Scaffold:**
- API: `rope_freqs`, `rope_sincos`, `rope` in `include/llmberry/kernels.h`
- Stub + shape checks: `kernels/rope.cpp` (throws `logic_error` until filled in)
- Sequential `rope_sincos` and convenience `rope(x, out, offset)` are glue — they call the three TODOs
- Tests: `tests/test_rope.cpp` (independent C++ reference, Llama / GPT-NeoX `rotate_half`)
- Python ref: `python/verify_ops.py` `rope_freqs()`, `rope_sincos()`, `rope()`

**Implement** (delete each `throw` in `kernels/rope.cpp`):
1. `rope_freqs` — `inv_freq[i] = 1 / theta^(2i / dim)`
2. `rope_sincos(inv_freq, positions, cos, sin)` — concat(freqs, freqs) then cos/sin
3. `rope(x, cos, sin, out)` — `y = x * cos + rotate_half(x) * sin`

Layout: `x` is `[seq, ..., head_dim]` (dim 0 = sequence, last dim even). Decode: seq=1 with `position_offset` = cache length. Default theta = 10000.

**Verify:**

```bash
make test-rope
python3 python/verify_ops.py
```

---

## 06 — Attention

**Status:** Complete

**Completed:**
- Numerically stable last-dim `softmax` (`kernels/softmax.cpp`)
- Scaled dot-product attention: QK → `1/sqrt(d)` → optional causal mask → softmax → AV
- `attention_qk` / `attention_av` via per-head `matrix(h)` + `matmul` (`Kᵀ` is a stride view)
- Causal mask: PyTorch `is_causal` bottom-right (`-inf` when `j + seq_q > i + seq_k`)
- Tensor helpers: `transpose()` (last two dims, no copy), `matrix(i)` (2-D plane view)
- `matmul` indexes with `strides[0]` and `strides[1]` so transposed `B` is valid
- C++ tests vs independent SDPA reference; NumPy/PyTorch ref in `python/verify_ops.py`

**Tests:**
- `test_attention`: 36/36 passing (`make test-attention`)

**Notes:**
- Layout is `[..., seq, dim]` (2-D single head, 3-D `[n_heads, seq, d]`, 4-D batched).
- Default scale is `1/sqrt(head_dim)` (Llama). `scale < 0` selects that default.
- Glue allocates `scores` and `weights`; no KV cache yet (checkpoint 13).
- GQA (`n_q != n_kv`) is checkpoint 07.

**Verify:**

```bash
make test-attention
python3 python/verify_ops.py
```

---

## 07 — Grouped-Query Attention

**Status:** Not started

**Verify:**

```bash
make build
./build/tests/test_attention
```

---

## 08 — SwiGLU MLP

**Status:** Not started

**Verify:**

```bash
make build
make test-matmul
```

(Add a dedicated MLP test binary when the op exists.)

---

## 09 — Transformer Block

**Status:** Not started

**Verify:**

```bash
make build
# layer-level gtest once added, e.g. ./build/tests/test_transformer
```

---

## 10 — Load Real Model Weights

**Status:** Not started

**Verify:**

```bash
make build
python3 python/convert_weights.py
./build/llmberry
```

---

## 11 — Full Model Forward Pass

**Status:** Not started

**Verify:**

```bash
make build
python3 python/verify_logits.py
```

---

## 12 — Autoregressive Text Generation

**Status:** Not started

**Verify:**

```bash
make build
make run
```

---

## 13 — KV Cache

**Status:** Not started

**Verify:**

```bash
make build
./build/benchmarks/benchmark_decode
```

Compare cached vs uncached decode latency.

---

## 14 — Separate Prefill and Decode

**Status:** Not started

**Verify:**

```bash
make build
./build/benchmarks/benchmark_decode
```

---

## 15 — Detailed Profiling

**Status:** Not started

**Verify:**

```bash
make build
# profile the decode/prefill binaries from 13–14
```

---

## 16 — Memory and Allocation Optimization

**Status:** Not started

**Verify:**

```bash
make build
./build/benchmarks/benchmark_decode
```

---

## 17 — CPU Kernel Optimization

**Status:** Not started

**Verify:**

```bash
make build
make test-matmul
./build/benchmarks/benchmark_matmul
./build/benchmarks/benchmark_matmul --large
```

Compare against the naive numbers in the benchmark log below.

---

## 18 — Weight Quantization

**Status:** Not started

**Verify:**

```bash
make build
python3 python/verify_logits.py
```

---

## 19 — Batching and Request Scheduling

**Status:** Not started

**Verify:**

```bash
make build
make run
```

---

## 20 — Paged KV Cache and Final Performance Study

**Status:** Not started

**Verify:**

```bash
make build
./build/benchmarks/benchmark_decode
make benchmark
```

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
