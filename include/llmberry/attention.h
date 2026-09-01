#pragma once

#include "llmberry/tensor.h"

namespace llmberry {

/// Scaled dot-product attention (Phase 1 checkpoint 06).
///
///   scores = Q K^T / sqrt(d)
///   if causal: scores[..., i, j] = -inf  when  j > i + seq_k - seq_q
///   out    = softmax(scores) @ V
///
/// Shapes (row-major; last two dims are sequence and channel):
///   q:   [..., seq_q, d]
///   k:   [..., seq_k, d]
///   v:   [..., seq_k, d_v]
///   out: [..., seq_q, d_v]
///
/// Leading dims are independent heads (and optional batch): e.g. [n_heads, seq, d]
/// or [batch, n_heads, seq, d]. Leading dims of q, k, and v must match.
/// Grouped-query attention (n_q != n_kv) is checkpoint 07.
///
/// Causal masking uses PyTorch `is_causal` bottom-right alignment, so decode
/// with seq_q=1, seq_k=T attends to every cached key.
/// `scale < 0` (the default) means 1/sqrt(d).

void attention_qk(const Tensor<float>& q, const Tensor<float>& k, Tensor<float>& scores);

/// In-place: scores[i] *= scale. Implemented — used by the attention glue.
void attention_scale(Tensor<float>& scores, float scale);

void attention_causal_mask(Tensor<float>& scores);

void attention_av(const Tensor<float>& attn, const Tensor<float>& v, Tensor<float>& out);

void attention(const Tensor<float>& q,
               const Tensor<float>& k,
               const Tensor<float>& v,
               Tensor<float>& out,
               bool causal = true,
               float scale = -1.0f);
Tensor<float> attention(const Tensor<float>& q,
                        const Tensor<float>& k,
                        const Tensor<float>& v,
                        bool causal = true,
                        float scale = -1.0f);

}  // namespace llmberry
