#include "llmberry/attention.h"
#include "llmberry/ensure.h"
#include "llmberry/kernels.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llmberry {
namespace {

std::vector<size_t> scores_shape_from(const Tensor<float>& q, const Tensor<float>& k) {
    std::vector<size_t> shape = q.shape();
    shape.back() = k.shape()[k.ndim() - 2];
    return shape;
}

std::vector<size_t> output_shape_from(const Tensor<float>& q, const Tensor<float>& v) {
    std::vector<size_t> shape = q.shape();
    shape.back() = v.shape().back();
    return shape;
}

void ensure_same_leading_dims(const Tensor<float>& a, const Tensor<float>& b, const char* msg) {
    ENSURE(a.ndim() == b.ndim(), std::invalid_argument, msg);
    for (size_t i = 0; i + 2 < a.ndim(); ++i) {
        ENSURE(a.shape()[i] == b.shape()[i], std::invalid_argument, msg);
    }
}

void check_qk(const Tensor<float>& q, const Tensor<float>& k, const Tensor<float>& scores) {
    ENSURE(!q.empty() && !k.empty(), std::invalid_argument, "attention_qk: q and k cannot be empty");
    ENSURE(q.ndim() >= 2 && k.ndim() >= 2, std::invalid_argument,
           "attention_qk: q and k must have shape [..., seq, d]");
    ENSURE(q.shape().back() == k.shape().back(), std::invalid_argument,
           "attention_qk: q and k head_dim must match");
    ensure_same_leading_dims(q, k, "attention_qk: leading dims of q and k must match");
    ENSURE(scores.shape() == scores_shape_from(q, k), std::invalid_argument,
           "attention_qk: scores must have shape [..., seq_q, seq_k]");
}

void check_av(const Tensor<float>& attn, const Tensor<float>& v, const Tensor<float>& out) {
    ENSURE(!attn.empty() && !v.empty(), std::invalid_argument,
           "attention_av: attn and v cannot be empty");
    ENSURE(attn.ndim() >= 2 && v.ndim() >= 2, std::invalid_argument,
           "attention_av: attn and v must have rank >= 2");
    ENSURE(attn.shape().back() == v.shape()[v.ndim() - 2], std::invalid_argument,
           "attention_av: attn seq_k must match v seq_k");
    ensure_same_leading_dims(attn, v, "attention_av: leading dims of attn and v must match");
    ENSURE(out.shape() == output_shape_from(attn, v), std::invalid_argument,
           "attention_av: out must have shape [..., seq_q, d_v]");
}

void check_attention(const Tensor<float>& q,
                     const Tensor<float>& k,
                     const Tensor<float>& v,
                     const Tensor<float>& out) {
    ENSURE(!q.empty() && !k.empty() && !v.empty(), std::invalid_argument,
           "attention: q, k, v cannot be empty");
    ENSURE(q.ndim() >= 2 && k.ndim() >= 2 && v.ndim() >= 2, std::invalid_argument,
           "attention: q, k, v must have shape [..., seq, dim]");
    ENSURE(q.shape().back() == k.shape().back(), std::invalid_argument,
           "attention: q and k head_dim must match");
    ENSURE(k.shape()[k.ndim() - 2] == v.shape()[v.ndim() - 2], std::invalid_argument,
           "attention: k and v seq_k must match");
    ensure_same_leading_dims(q, k, "attention: leading dims of q and k must match");
    ensure_same_leading_dims(q, v, "attention: leading dims of q and v must match");
    ENSURE(out.shape() == output_shape_from(q, v), std::invalid_argument,
           "attention: out must have shape [..., seq_q, d_v]");
}

}  // namespace

void attention_qk(const Tensor<float>& q, const Tensor<float>& k, Tensor<float>& scores) {
    check_qk(q, k, scores);

    const size_t seq_q = q.shape()[q.ndim() - 2];
    const size_t d = q.shape().back();
    const size_t seq_k = k.shape()[k.ndim() - 2];
    const size_t matrix_size = seq_q * d;
    const size_t n_heads = q.size() / matrix_size;

    for (size_t h = 0; h < n_heads; ++h) {
       Tensor<float> out_h = scores.matrix(h);
       matmul(q.matrix(h), k.matrix(h).transpose(), out_h);
    }

}

void attention_scale(Tensor<float>& scores, float scale) {
    ENSURE(!scores.empty(), std::invalid_argument, "attention_scale: scores cannot be empty");
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] *= scale;
    }
}

void attention_causal_mask(Tensor<float>& scores) {
    ENSURE(!scores.empty(), std::invalid_argument, "attention_causal_mask: scores cannot be empty");
    ENSURE(scores.ndim() >= 2, std::invalid_argument,
           "attention_causal_mask: scores must have shape [..., seq_q, seq_k]");

    const size_t seq_q = scores.shape()[scores.ndim() - 2];
    const size_t seq_k = scores.shape().back();
    const size_t n_heads = scores.size() / (seq_q * seq_k);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // PyTorch is_causal, bottom-right aligned:
    //   scores[..., i, j] = -inf  when  j > i + seq_k - seq_q
    // Equivalent unsigned test: j + seq_q > i + seq_k
    //
    // Square prefill (seq_q == seq_k): lower-triangular (query i cannot see key j > i).
    // Decode (seq_q == 1): no keys are masked.

    for (size_t h = 0; h < n_heads; ++h) {
        Tensor<float> plane = scores.matrix(h);
        for (size_t i = 0; i < seq_q; ++i) {
            for (size_t j = 0; j < seq_k; ++j) {
                if (j + seq_q > i + seq_k) {
                    plane.at({i, j}) = neg_inf;
                }
            }
        }
    }
}

void attention_av(const Tensor<float>& attn, const Tensor<float>& v, Tensor<float>& out) {
    check_av(attn, v, out);

    const size_t seq_q = attn.shape()[attn.ndim() - 2];
    const size_t seq_k = attn.shape().back();
    const size_t d_v = v.shape().back();
    const size_t n_heads = v.size() / (seq_k * d_v);

       for(size_t h=0; h < n_heads; h++){
              Tensor<float> out_tensor = out.matrix(h);
              matmul(attn.matrix(h), v.matrix(h), out_tensor);
       }
}

void attention(const Tensor<float>& q,
               const Tensor<float>& k,
               const Tensor<float>& v,
               Tensor<float>& out,
               bool causal,
               float scale) {
    check_attention(q, k, v, out);

    const size_t d = q.shape().back();
    const float s = (scale < 0.0f) ? 1.0f / std::sqrt(static_cast<float>(d)) : scale;

    Tensor<float> scores(scores_shape_from(q, k));
    attention_qk(q, k, scores);
    attention_scale(scores, s);
    if (causal) {
        attention_causal_mask(scores);
    }
    Tensor<float> weights(scores.shape());
    softmax(scores, weights);
    attention_av(weights, v, out);
}

Tensor<float> attention(const Tensor<float>& q,
                        const Tensor<float>& k,
                        const Tensor<float>& v,
                        bool causal,
                        float scale) {
    ENSURE(!q.empty(), std::invalid_argument, "attention: q cannot be empty");
    ENSURE(q.ndim() >= 2 && v.ndim() >= 2, std::invalid_argument,
           "attention: q, k, v must have shape [..., seq, dim]");
    Tensor<float> out(output_shape_from(q, v));
    attention(q, k, v, out, causal, scale);
    return out;
}

}  // namespace llmberry
