#pragma once

#include "llmberry/ensure.h"
#include "llmberry/tensor.h"

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

}  // namespace llmberry
