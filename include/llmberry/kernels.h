#pragma once

#include "llmberry/ensure.h"
#include "llmberry/tensor.h"

#include <vector>

namespace llmberry {

// Naive CPU reference kernels (Phase 1 checkpoint 02).
// Correctness first; later checkpoints replace these with optimized versions.

/// Matrix multiply: C = A @ B.
/// A is [M, K], B is [K, N], C is [M, N]. Row-major.
void matmul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c);

/// Matrix-vector: y = A @ x.
/// A is [M, N], x is [N], y is [M].
void gemv(const Tensor<float>& a, const Tensor<float>& x, Tensor<float>& y);

/// Elementwise addition: out[i] = a[i] + b[i]. Matching shapes; writes into `out`.
void add(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out);
/// Allocating wrapper: returns a new tensor with shape `a.shape()`.
Tensor<float> add(const Tensor<float>& a, const Tensor<float>& b);

/// Elementwise multiplication: out[i] = a[i] * b[i]. Matching shapes; writes into `out`.
void mul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out);
/// Allocating wrapper: returns a new tensor with shape `a.shape()`.
Tensor<float> mul(const Tensor<float>& a, const Tensor<float>& b);

/// Dot product of two 1-D tensors of equal length.
float dot(const Tensor<float>& a, const Tensor<float>& b);

/// Sum of all elements.
float reduce_sum(const Tensor<float>& x);

/// Mean of all elements.
float reduce_mean(const Tensor<float>& x);

/// SiLU: out[i] = x[i] * sigmoid(x[i]), where sigmoid(z) = 1 / (1 + exp(-z)).
void silu(const Tensor<float>& x, Tensor<float>& out);
Tensor<float> silu(const Tensor<float>& x);

/// GELU (exact / erf): out[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2))).
void gelu(const Tensor<float>& x, Tensor<float>& out);
Tensor<float> gelu(const Tensor<float>& x);

/// RMSNorm over the last dimension (Llama / Phase 1 checkpoint 04).
///
///   y = x / sqrt(mean(x^2) + eps)  ⊙  weight
///
/// `x` and `out` have shape [..., D]; `weight` is [D]. Each vector along the
/// last axis is normalized independently, then scaled by `weight`.
/// Default `eps` matches Hugging Face LlamaRMSNorm (1e-6).
void rmsnorm(const Tensor<float>& x,
             const Tensor<float>& weight,
             Tensor<float>& out,
             float eps = 1e-6f);
Tensor<float> rmsnorm(const Tensor<float>& x,
                      const Tensor<float>& weight,
                      float eps = 1e-6f);

/// Rotary positional embeddings (Llama / GPT-NeoX, Phase 1 checkpoint 05).
///
/// Frequencies (`inv_freq` has shape [head_dim/2], head_dim even):
///
///   inv_freq[i] = 1 / theta^(2i / head_dim)    i = 0 .. head_dim/2 - 1
///
/// Default `theta` is 10000 (Llama 1/2). Llama 3 uses 500000.
///
/// Sine/cosine tables (`cos`, `sin` have shape [n_pos, head_dim]):
///
///   freqs[t, i] = positions[t] * inv_freq[i]
///   emb         = concat(freqs, freqs) along the last dim   (HF Llama)
///   cos, sin    = cos(emb), sin(emb)
///
/// Apply (`x` / `out` have shape [seq, ..., head_dim]; dim 0 is sequence):
///
///   y = x * cos + rotate_half(x) * sin
///   rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])
///
/// Prefill uses seq > 1; single-token decode uses shape [1, ..., head_dim]
/// with `position_offset` equal to the KV-cache length.

void rope_freqs(size_t dim, Tensor<float>& inv_freq, float theta = 10000.0f);
Tensor<float> rope_freqs(size_t dim, float theta = 10000.0f);

/// Sequential positions [position_offset, position_offset + seq_len).
/// `cos` and `sin` must have shape [seq_len, dim].
void rope_sincos(size_t seq_len,
                 size_t dim,
                 Tensor<float>& cos,
                 Tensor<float>& sin,
                 size_t position_offset = 0,
                 float theta = 10000.0f);

/// Arbitrary token positions (decode, packed sequences).
/// `inv_freq` is [dim/2]; `cos` / `sin` are [positions.size(), dim].
void rope_sincos(const Tensor<float>& inv_freq,
                 const std::vector<size_t>& positions,
                 Tensor<float>& cos,
                 Tensor<float>& sin);

void rope(const Tensor<float>& x,
          const Tensor<float>& cos,
          const Tensor<float>& sin,
          Tensor<float>& out);
Tensor<float> rope(const Tensor<float>& x,
                   const Tensor<float>& cos,
                   const Tensor<float>& sin);

/// Convenience: build tables for sequential positions and apply.
void rope(const Tensor<float>& x,
          Tensor<float>& out,
          size_t position_offset = 0,
          float theta = 10000.0f);
Tensor<float> rope(const Tensor<float>& x,
                   size_t position_offset = 0,
                   float theta = 10000.0f);

/// Numerically stable softmax over the last dimension (Phase 1 checkpoint 06).
///
///   m = max(x[..., :])
///   y = exp(x - m) / sum(exp(x - m))
///
/// `x` and `out` have matching shape [..., D]. Each vector along the last
/// axis is normalized independently so that it sums to 1.
/// A `-inf` logit becomes probability 0 (used by causal attention masks).
void softmax(const Tensor<float>& x, Tensor<float>& out);
Tensor<float> softmax(const Tensor<float>& x);
}  // namespace llmberry
