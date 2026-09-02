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
| `make test-rmsnorm` | Checkpoint 04 RMSNorm tests |
| `make test-rope` | Checkpoint 05 RoPE tests |
| `make test-attention` | Checkpoint 06–07 attention tests (softmax + SDPA + GQA) |
| `make test-mlp` | Checkpoint 08 SwiGLU MLP tests |
| `make verify-ops` | C++ `dump_ops` vs NumPy (PyTorch too if installed) |
| `make verify` | Build (if needed) + full test suite (includes `verify_ops`) |

---

## Checkpoint 01 — Tensor

**Gate:** all tests in `tests/test_tensor.cpp` pass (19 tests).

| Group | Filter | Count | Covers |
|-------|--------|-------|--------|
| Construction | `TensorConstruction.*` | 3 | shape, strides, size, nbytes, dtype, zero-dim rejection |
| Dtype | `TensorDtype.*` | 1 | `dtype_of` / `Tensor::dtype()` |
| Factories | `TensorFactories.*` | 2 | `zeros`, `ones` |
| Flat indexing | `TensorIndexing.Flat*` | 2 | `operator[]`, bounds |
| Multi-dim | `TensorIndexing.At*` | 4 | `at()`, rank / bounds errors |
| Data pointer | `TensorData.*` | 1 | `data()` row-major layout |
| Views | `TensorView.*` | 5 | reshape view, `row()` slice, invalid reshape, external buffer |
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

**Also:** `python/verify_ops.py` runs `dump_ops` and checks C++ vs NumPy (and PyTorch if installed).

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

## Checkpoint 04 — RMSNorm

**Gate:** all tests in `tests/test_rmsnorm.cpp` pass (`make test-rmsnorm`).

| Group | Filter | Covers |
|-------|--------|--------|
| Known values | `RMSNorm.OnesWeightConstantVector`, `ZeroInputIsZero` | hand-checkable output |
| Weight / layout | `WeightScales*`, `LastDim*` | per-row last-dim reduction, affine scale |
| Reference | `MatchesIndependentReference` | 1-D / 2-D / 3-D vs C++ ref |
| Contract | `Overwrites*`, `LeavesInputs*`, `AllocatingWrapper` | in-place out, no input mutation |
| Stability | `DefaultEpsMatchesLlama`, `SmallValuesStayFinite` | eps=1e-6, no NaN on tiny x |
| Errors | `ShapeMismatchThrows`, `EmptyThrows` | weight rank/size, out shape |

**Also:** `python/verify_ops.py` `rmsnorm()` matches NumPy / Hugging Face LlamaRMSNorm.

```bash
make test-rmsnorm
python3 python/verify_ops.py
```

---

## Checkpoint 05 — RoPE

**Gate:** all tests in `tests/test_rope.cpp` pass (`make test-rope`).

| Group | Filter | Covers |
|-------|--------|--------|
| Frequencies | `RoPEFreqs.*` | `inv_freq[i] = 1 / theta^(2i/dim)`, Llama-3 theta, odd dim |
| Sin/cos tables | `RoPESinCos.*` | position 0 identity, concat layout, unit circle |
| Known values | `RoPE.PositionZeroIsIdentity`, `KnownRotationDim2` | hand-checkable rotation |
| Invariants | `PreservesNorm`, `HeadsSharePosition` | orthogonal rotation, broadcast over heads |
| Reference | `MatchesIndependentReference`, `PositionOffset`, `ArbitraryPositions` | Llama/NeoX apply, decode offset |
| Contract | `Overwrites*`, `LeavesInputs*`, `AllocatingWrapper`, `DefaultThetaIs10000` | in-place out, theta=10000 |
| Layout | `ContiguousReshapeView` | view sharing storage |
| Errors | `ShapeMismatchThrows`, `EmptyThrows`, `OddHeadDimThrows` | rank, shapes, odd head_dim |

**Also:** `python/verify_ops.py` `rope()` / `rope_freqs()` / `rope_sincos()` match Hugging Face Llama (rotate_half).

```bash
make test-rope
python3 python/verify_ops.py
```

---

## Checkpoint 06 — Attention

**Gate:** all tests in `tests/test_attention.cpp` pass (`make test-attention`).

| Group | Filter | Covers |
|-------|--------|--------|
| Softmax known values | `Softmax.KnownValues1D`, `UniformWhenEqual`, `NegInfBecomesZero` | hand-checkable output, causal `-inf` |
| Softmax invariants | `SumsToOne`, `NumericallyStableLargeValues`, `LastDimIndependent` | last-dim, no overflow |
| Softmax reference | `Softmax.MatchesIndependentReference` | 1-D / 2-D / 3-D / 4-D vs C++ ref |
| Softmax contract | `Overwrites*`, `LeavesInput*`, `AllocatingWrapper` | in-place out, no input mutation |
| Softmax errors | `ShapeMismatchThrows`, `EmptyThrows` | out shape, empty |
| QK^T | `AttentionQK.*` | `Q @ K^T`, multi-head, shape errors |
| Causal mask | `AttentionMask.*` | lower-triangular, decode, bottom-right, heads |
| Attention × V | `AttentionAV.*` | one-hot gather, batched GEMM, shape errors |
| SDPA | `Attention.CausalLowerTriangular*`, `NonCausalDiffers*`, `MultiHead*` | causal, heads independent |
| SDPA reference | `MatchesIndependentReference`, `DecodeSingleQuery`, `CustomScale`, `DefaultScale*` | PyTorch SDPA, `1/sqrt(d)` |
| SDPA contract | `Overwrites*`, `LeavesInputs*`, `AllocatingWrapper`, `ScaleHelper`, `ContiguousReshapeView` | glue, views |
| SDPA errors | `ShapeMismatchThrows`, `EmptyThrows` | rank, d mismatch, empty |

Shape / empty tests pass on the scaffold (checks run before the TODOs). Compute tests fail with `logic_error` until you fill in the four `throw`s.

**Also:** `python/verify_ops.py` `softmax()` / `attention()` match NumPy / PyTorch SDPA.

```bash
make test-attention
python3 python/verify_ops.py
```

## Checkpoint 07 — Grouped-Query Attention

**Gate:** all tests in `tests/test_attention.cpp` pass (`make test-attention`), including `AttentionGqa.*`.

| Group | Filter | Covers |
|-------|--------|--------|
| GQA kernels | `AttentionQK.Gqa*`, `AttentionAV.Gqa*` | shared KV heads, batched QK |
| GQA prefill | `AttentionGqa.MatchesExpandedMhaPrefill`, `Mqa`, `BatchedPrefill` | repeat-interleave ≡ MHA, MQA |
| GQA decode | `DecodeSingleQuery`, `Prefill*EqualsDecode`, `PrefillThenMultiStepDecode`, `BatchedDecode` | `seq_q=1` vs growing K/V prefix |
| GQA extras | `GroupSharesKvHead`, `ChunkPrefillBottomRight`, `DifferentValueDim`, `CustomScale` | grouping, `d_v != d` |
| GQA contract | `AllocatingWrapper`, `LeavesInputsUnchanged`, `NqEqualsNkvIsMha` | MHA still works |
| GQA errors | `NqNotDivisible*`, `KvHeadMismatch*`, `BatchMismatch*`, `RankMismatch*` | invalid head layout |

**Also:** `python/verify_ops.py` `attention()` repeat-interleaves K/V when `n_q != n_kv`.

```bash
make test-attention
python3 python/verify_ops.py
```

---

## Checkpoint 08 — SwiGLU MLP

**Gate:** all tests in `tests/test_mlp.cpp` pass (`make test-mlp`).

| Group | Filter | Covers |
|-------|--------|--------|
| Known values | `MLP.KnownValuesTiny` | hand-checkable SiLU + projections |
| Algebra | `ZeroGateIsZero`, `ZeroDownIsZero` | SiLU(0)·up = 0; down = 0 |
| Token independence | `PrefillEqualsPerTokenDecode` | `[seq, H]` ≡ per-row `[H]` |
| Reference | `MatchesIndependentReference` | 1-D / 2-D / 3-D vs C++ ref, `I ≠ kH` |
| Contract | `Overwrites*`, `LeavesInputs*`, `AllocatingWrapper` | in-place out, no input mutation |
| Layout | `ContiguousReshapeView` | view sharing storage |
| Errors | `ShapeMismatchThrows`, `EmptyThrows` | ranks, `[H,I]` / `[I,H]`, empty |

**Also:** `python/verify_ops.py` `mlp()` matches NumPy; HF Linear weights are `[out, in]` — pass `W.T`. `dump_ops` includes `mlp` and `mlp_batched`.

```bash
make test-mlp
python3 python/verify_ops.py
```

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
