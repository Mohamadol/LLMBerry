# LLMBerry

**LLMBerry** is a lightweight C++ runtime for LLM inference and decode optimization — a Llama-compatible inference engine written from scratch to explore and optimize the end-to-end LLM inference pipeline.

The project starts with a simple, correct CPU implementation and gradually evolves into an optimized inference engine with KV caching, specialized prefill/decode paths, quantization, continuous batching, and paged KV-cache management.

The main goal is not to build another full-featured production framework. Instead, LLMBerry is designed as a hands-on study of:

* LLM inference internals
* C++ systems programming
* CPU performance optimization
* memory management
* inference profiling
* KV-cache design
* quantization
* request scheduling
* throughput and latency optimization

Python is used only for model conversion, validation, benchmarking, and visualization. The inference runtime itself is implemented primarily in C++.

---

## Project Goals

LLMBerry aims to answer questions such as:

* Where is time spent during LLM inference?
* How do prefill and decode differ computationally?
* How much does KV caching improve autoregressive decoding?
* When does inference become memory-bandwidth bound?
* How should tensors and KV caches be laid out in memory?
* How much performance can SIMD and multithreading provide?
* How does quantization affect throughput, memory usage, and accuracy?
* How does batching affect throughput and latency?
* How can paged KV-cache allocation improve memory utilization?

The development process follows a simple methodology:

> **Implement → Validate → Profile → Optimize → Measure**

---

# Roadmap

## Phase 1 — Build a Correct Inference Engine

### [ ] 01 — Project Setup and Tensor Abstraction

Set up the initial C++ project.

Tasks:

* Configure CMake
* Create source/include/test directories
* Add unit testing
* Add benchmark infrastructure
* Implement a lightweight `Tensor` abstraction
* Support:

  * shape
  * strides
  * dtype
  * memory allocation
  * indexing
  * tensor views

Example:

```cpp
Tensor<float> x({2, 4, 8});
```

**Checkpoint:** Basic tensor operations work correctly and are covered by tests.

---

### [x] 02 — Basic CPU Kernels

Implement naive reference kernels in C++.

Initial operations:

* matrix multiplication
* matrix-vector multiplication
* elementwise multiplication
* elementwise addition
* dot product
* reductions
* SiLU activation

Performance is not the priority yet.

These kernels establish a correctness baseline against which future optimized implementations can be compared.

**Checkpoint:** C++ kernels match NumPy/PyTorch reference outputs.

---

### [ ] 03 — Benchmarking Infrastructure

Create a reusable C++ benchmarking framework.

Measure:

* execution latency
* throughput
* FLOP/s
* memory bandwidth where applicable

Use Python scripts to generate benchmark plots.

Example:

```text
Matrix: 4096 × 4096
Threads: 1

Latency:        12.4 ms
Throughput:     2.7 GFLOP/s
```

**Checkpoint:** Every future optimization can be measured reproducibly.

---

### [x] 04 — RMSNorm

Implement the RMSNorm operation used by Llama models.

$$
y =
\frac{x}
{\sqrt{\operatorname{mean}(x^2)+\epsilon}}
\odot w
$$

Tasks:

* implement forward pass
* handle numerical stability
* compare against PyTorch

**Checkpoint:** RMSNorm output matches the reference implementation within numerical tolerance.

---

### [ ] 05 — Rotary Positional Embeddings

Implement Rotary Positional Embeddings (RoPE).

Tasks:

* generate RoPE frequencies
* precompute sine/cosine tables
* apply rotations to query and key tensors
* support arbitrary token positions

Conceptually:

$$
Q' = \operatorname{RoPE}(Q)
$$

$$
K' = \operatorname{RoPE}(K)
$$

**Checkpoint:** C++ RoPE matches a Python reference implementation.

---

### [ ] 06 — Attention

Implement scaled dot-product attention.

$$
A =
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)
$$

$$
O = AV
$$

Tasks:

* QK multiplication
* scaling
* causal masking
* numerically stable softmax
* attention × V

Start with standard multi-head attention.

**Checkpoint:** Attention output matches PyTorch.

---

### [ ] 07 — Grouped-Query Attention

Extend the attention implementation to support **Grouped-Query Attention (GQA)** used by modern Llama models.

Support:

$$
N_Q > N_{KV}
$$

where multiple query heads share the same key/value heads.

Study the effect of GQA on KV-cache memory requirements.

**Checkpoint:** GQA output matches the reference model.

---

### [ ] 08 — SwiGLU MLP

Implement the Llama feed-forward network.

$$
h =
\operatorname{SiLU}(xW_g)
\odot
(xW_u)
$$

followed by:

$$
y = hW_d
$$

Tasks:

* gate projection
* up projection
* SiLU
* elementwise multiplication
* down projection

**Checkpoint:** MLP output matches PyTorch.

---

### [ ] 09 — Transformer Block

Combine the implemented components into one complete Llama Transformer layer.

Pipeline:

```text
Input
  │
  ▼
RMSNorm
  │
  ▼
Attention
  │
  ▼
Residual
  │
  ▼
RMSNorm
  │
  ▼
SwiGLU MLP
  │
  ▼
Residual
```

**Checkpoint:** A complete Transformer block matches the corresponding reference layer.

---

### [ ] 10 — Load Real Model Weights

Load weights from a real Llama-compatible model.

Use Python to:

1. read Hugging Face / Safetensors weights
2. convert them into a simple LLMBerry binary format

Use C++ to:

1. parse model metadata
2. allocate model tensors
3. load weights
4. map tensors to Transformer layers

Start with a small Llama-compatible model for fast development.

**Milestone:** LLMBerry can load real pretrained model weights.

---

# Phase 2 — Build an Actual LLM Runtime

### [ ] 11 — Full Model Forward Pass

Stack Transformer layers into a complete model.

Implement:

* token embeddings
* all Transformer blocks
* final RMSNorm
* language-model head
* logits generation

Pipeline:

```text
Token IDs
   │
   ▼
Embeddings
   │
   ▼
Transformer × N
   │
   ▼
RMSNorm
   │
   ▼
LM Head
   │
   ▼
Logits
```

Compare logits against Hugging Face.

**Checkpoint:** LLMBerry produces numerically consistent logits for real prompts.

---

### [ ] 12 — Autoregressive Text Generation

Implement token-by-token generation.

Start with:

* greedy decoding

Then add:

* temperature sampling
* top-k sampling
* top-p sampling

Tokenization may initially be performed in Python or through an existing tokenizer library.

Example:

```bash
./llmberry \
    --model model.bin \
    --prompt "The capital of France is"
```

Expected result:

```text
The capital of France is Paris...
```

**Milestone:** LLMBerry can generate text using a real pretrained LLM.

---

### [ ] 13 — KV Cache

Implement KV caching for autoregressive inference.

Without a cache, previously generated tokens require repeated key/value computation.

With a cache:

```text
Token 1 → K₁ V₁
Token 2 → K₂ V₂
Token 3 → K₃ V₃

KV Cache:
[K₁ K₂ K₃ ...]
[V₁ V₂ V₃ ...]
```

Benchmark decode latency across context lengths.

Example:

| Context Length | No KV Cache | KV Cache |
| -------------: | ----------: | -------: |
|            128 |         TBD |      TBD |
|            512 |         TBD |      TBD |
|           2048 |         TBD |      TBD |
|           4096 |         TBD |      TBD |

**Milestone:** Demonstrate the performance benefit of KV caching.

---

### [ ] 14 — Separate Prefill and Decode

Explicitly separate the two major inference workloads.

```cpp
model.prefill(prompt_tokens);

while (!finished) {
    token = model.decode(previous_token);
}
```

### Prefill

Processes many prompt tokens simultaneously.

Typical characteristics:

* large matrix multiplications
* relatively compute intensive
* high parallelism

### Decode

Processes one new token per sequence at each step.

Typical characteristics:

* small matrix dimensions
* frequent weight reads
* KV-cache accesses
* often memory-bandwidth limited

Benchmark the two independently.

**Milestone:** LLMBerry now has an inference architecture resembling real LLM runtimes.

---

# Phase 3 — Optimize the Inference Pipeline

### [ ] 15 — Detailed Profiling

Instrument every major decode component.

Measure:

* RMSNorm
* QKV projection
* RoPE
* QK
* softmax
* AV
* output projection
* gate projection
* up projection
* down projection
* sampling
* memory operations

Generate latency breakdown plots using Python.

Example:

```text
Decode latency

QKV projection     ████████████
Attention          ████
Output projection  █████
MLP                ███████████████
RMSNorm            ██
Other              █
```

From this point forward:

> **Do not optimize based on intuition. Profile first.**

**Milestone:** Identify the true decode bottlenecks.

---

### [ ] 16 — Memory and Allocation Optimization

Remove unnecessary memory operations from the inference hot path.

Explore:

* preallocated workspaces
* reusable buffers
* tensor views
* cache-friendly tensor layouts
* aligned allocations
* avoiding temporary tensors
* reducing memory copies
* improving KV-cache layout
* minimizing allocation during decode

Benchmark every optimization separately.

Example:

| Optimization       | Decode Latency |
| ------------------ | -------------: |
| Baseline           |            TBD |
| Remove allocations |            TBD |
| Buffer reuse       |            TBD |
| Improved layout    |            TBD |

**Checkpoint:** Decode performs minimal dynamic memory allocation.

---

### [ ] 17 — CPU Kernel Optimization

Optimize the most important kernels.

Explore:

* loop reordering
* cache blocking
* loop tiling
* SIMD
* AVX2
* AVX-512 where supported
* multithreading
* vectorized reductions
* fused operations

Pay particular attention to the difference between:

```text
Prefill → GEMM-heavy

Decode → often GEMV / small-GEMM-heavy
```

Compare custom kernels against optimized BLAS implementations where appropriate.

Measure:

* latency
* FLOP/s
* memory bandwidth
* scaling with thread count

Example experiment:

| Threads | Decode tok/s |
| ------: | -----------: |
|       1 |          TBD |
|       2 |          TBD |
|       4 |          TBD |
|       8 |          TBD |
|      16 |          TBD |

**Milestone:** Demonstrate measurable speedups from low-level C++ optimization.

---

### [ ] 18 — Weight Quantization

Implement weight-only quantization.

Start with:

```text
FP32
  ↓
FP16 / BF16
  ↓
INT8
```

Optional later extension:

```text
INT4
```

Python can perform offline weight conversion.

C++ performs quantized inference.

Study the tradeoff between:

* model memory
* memory bandwidth
* decode throughput
* latency
* output quality

Example:

| Precision | Model Memory | Decode tok/s | Accuracy |
| --------- | -----------: | -----------: | -------: |
| FP32      |          TBD |          TBD | Baseline |
| FP16      |          TBD |          TBD |      TBD |
| INT8      |          TBD |          TBD |      TBD |

**Milestone:** Demonstrate how reduced precision changes the inference bottleneck.

---

# Phase 4 — Production-Style Inference

### [ ] 19 — Batching and Request Scheduling

Extend LLMBerry from one sequence to multiple concurrent requests.

Introduce a request abstraction:

```cpp
struct Request {
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;

    KVCache kv_cache;

    RequestState state;
};
```

First implement:

### Static batching

All requests begin and execute together.

Then implement:

### Continuous batching

Completed requests leave the batch while new requests can enter.

Example:

```text
Step 1:
[A B C D]

Step 2:
[A B C D]

C finishes

Step 3:
[A B E D]

B finishes

Step 4:
[A F E D]
```

Measure:

* tokens/sec
* requests/sec
* time to first token (TTFT)
* inter-token latency (ITL)
* batch-size scaling

**Milestone:** LLMBerry supports multiple simultaneous generation requests.

---

### [ ] 20 — Paged KV Cache and Final Performance Study

Replace large contiguous per-request KV buffers with fixed-size blocks.

Instead of:

```text
Request A:
[------------------------ maximum context ------------------------]
```

use:

```text
Physical KV Blocks

Block 0
Block 1
Block 2
Block 3
...
```

Requests reference arbitrary blocks:

```text
Request A → [Block 7] [Block 2] [Block 19]

Request B → [Block 5] [Block 13]

Request C → [Block 8] [Block 4] [Block 21]
```

Implement:

* fixed-size KV blocks
* free-block pool
* block allocation
* block reclamation
* logical-to-physical block mapping

Measure:

* KV-cache utilization
* memory fragmentation
* maximum concurrent sequences
* throughput
* scheduling overhead

---

# Final Performance Study

The final evaluation should demonstrate how LLMBerry evolves from the original baseline.

Example:

```text
Naive C++ baseline
        │
        ▼
+ KV Cache
        │
        ▼
+ Prefill / Decode Specialization
        │
        ▼
+ Memory Optimization
        │
        ▼
+ SIMD + Multithreading
        │
        ▼
+ Quantization
        │
        ▼
+ Continuous Batching
        │
        ▼
+ Paged KV Cache
```

Report the contribution of each optimization independently.

Example:

| Configuration         | Decode tok/s | Speedup |
| --------------------- | -----------: | ------: |
| Baseline              |          TBD |   1.00× |
| + KV Cache            |          TBD |     TBD |
| + Memory Optimization |          TBD |     TBD |
| + Optimized Kernels   |          TBD |     TBD |
| + INT8                |          TBD |     TBD |
| Final                 |          TBD |     TBD |

The objective is not simply to report the final throughput.

The important question is:

> **Why did each optimization improve performance?**

---

# Performance Metrics

LLMBerry will track several metrics throughout development.

### Decode Throughput

$$
\text{Throughput}
=
\frac{\text{Generated Tokens}}
{\text{Execution Time}}
$$

Reported as:

```text
token/s
```

### Time to First Token

Time between receiving the prompt and generating the first output token.

### Inter-Token Latency

Time between successive generated tokens.

### Memory Usage

Measure:

* model weights
* KV cache
* temporary buffers
* peak runtime memory

### Scaling

Evaluate performance as a function of:

* context length
* batch size
* thread count
* model size
* numerical precision

---

# Repository Structure

```text
LLMBerry/
│
├── CMakeLists.txt
├── README.md
│
├── include/
│   ├── tensor.h
│   ├── model.h
│   ├── transformer.h
│   ├── attention.h
│   ├── kv_cache.h
│   ├── scheduler.h
│   ├── tokenizer.h
│   └── kernels.h
│
├── src/
│   ├── tensor.cpp
│   ├── model.cpp
│   ├── transformer.cpp
│   ├── attention.cpp
│   ├── kv_cache.cpp
│   ├── scheduler.cpp
│   ├── sampling.cpp
│   └── main.cpp
│
├── kernels/
│   ├── matmul.cpp
│   ├── gemv.cpp
│   ├── rmsnorm.cpp
│   ├── rope.cpp
│   ├── softmax.cpp
│   ├── silu.cpp
│   └── quantized_matmul.cpp
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_matmul.cpp
│   ├── test_rmsnorm.cpp
│   ├── test_rope.cpp
│   └── test_attention.cpp
│
├── benchmarks/
│   ├── benchmark_matmul.cpp
│   ├── benchmark_attention.cpp
│   └── benchmark_decode.cpp
│
└── python/
    ├── convert_weights.py
    ├── verify_ops.py
    ├── verify_logits.py
    ├── benchmark.py
    └── plot_results.py
```

---

# Development Philosophy

Each optimization should follow the same process.

### 1. Establish correctness

Validate C++ results against a trusted Python/PyTorch implementation.

### 2. Establish a baseline

Measure the unoptimized implementation.

### 3. Profile

Determine where execution time is actually spent.

### 4. Form a hypothesis

Example:

> Single-token decode appears memory-bandwidth limited because projection weights must be streamed from memory for every generated token.

### 5. Optimize

Modify the implementation based on the identified bottleneck.

### 6. Measure again

Report both absolute performance and relative speedup.

### 7. Explain the result

Document why the optimization succeeded—or why it did not.

---

# Major Milestones

### Milestone 1 — Checkpoint 10

Real model weights can be loaded into the C++ runtime.

**Demonstrates:**

* C++
* Transformer architecture
* tensor implementation
* numerical validation

---

### Milestone 2 — Checkpoint 14

Full autoregressive inference with KV caching and dedicated prefill/decode paths.

**Demonstrates:**

* LLM inference architecture
* KV caching
* memory management
* inference workloads

---

### Milestone 3 — Checkpoint 17

Profile-driven CPU optimization with SIMD, multithreading, and specialized kernels.

**Demonstrates:**

* systems programming
* profiling
* computer architecture
* low-level optimization

---

### Milestone 4 — Checkpoint 18

Quantized inference.

**Demonstrates:**

* numerical representation
* memory-bandwidth optimization
* quantized kernel implementation

---

### Milestone 5 — Checkpoint 20

Continuous batching and paged KV-cache management.

**Demonstrates:**

* inference serving
* scheduling
* memory management
* throughput/latency tradeoffs
* production-style LLM runtime concepts

---

# Possible Future Extensions

After completing the CPU runtime:

* CUDA backend
* custom CUDA decode kernels
* Triton comparison
* FlashAttention-style kernels
* fused attention
* fused RMSNorm
* fused MLP
* speculative decoding
* prefix caching
* tensor parallelism
* pipeline parallelism
* NUMA-aware execution
* INT4 quantization
* AWQ/GPTQ-style quantization
* CPU/GPU heterogeneous inference
* OpenAI-compatible inference server

---

# End Goal

By the end of the project, LLMBerry should provide a clear experimental history showing how an LLM inference engine progresses from a simple implementation to an optimized runtime.

The final project should demonstrate not only:

> **“I implemented Llama inference in C++.”**

but more importantly:

> **“I profiled an LLM inference pipeline, identified its compute and memory bottlenecks, implemented targeted systems-level optimizations, and quantitatively measured their effects on latency, throughput, and memory consumption.”**
