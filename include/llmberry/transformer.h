#pragma once

#include "llmberry/tensor.h"

namespace llmberry {

/// Llama decoder layer config (Phase 1 checkpoint 09).
///
///   hidden       = n_heads * head_dim
///   n_heads      % n_kv_heads == 0          (MHA, GQA, or MQA)
///   head_dim     even                       (RoPE)
struct TransformerConfig {
    size_t hidden = 0;
    size_t n_heads = 0;
    size_t n_kv_heads = 0;
    size_t intermediate = 0;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-6f;

    size_t head_dim() const { return n_heads == 0 ? 0 : hidden / n_heads; }
    size_t q_dim() const { return n_heads * head_dim(); }
    size_t kv_dim() const { return n_kv_heads * head_dim(); }
};

/// One Llama Transformer block. Owns layer dimensions and weights.
///
///   h = x + SelfAttn(RMSNorm(x))
///   y = h + MLP(RMSNorm(h))
///
/// Self-attention: QKV projections, RoPE on Q/K, causal GQA, output projection.
/// No bias (Hugging Face LlamaDecoderLayer). No KV cache (checkpoint 13).
///
/// Weight layout is compute-native (not Hugging Face Linear):
///   w_q:    [hidden, n_heads * head_dim]
///   w_k/v:  [hidden, n_kv_heads * head_dim]
///   w_o:    [n_heads * head_dim, hidden]
///   w_gate/up: [hidden, intermediate]
///   w_down: [intermediate, hidden]
///
/// Hugging Face stores Linear weights as [out, in]; transpose once at load
/// (checkpoint 10): W_cpp = W_hf.T.
///
/// Rank: x may be [hidden] (decode), [seq, hidden] (prefill), or
/// [batch, seq, hidden]. `position_offset` is the RoPE start (0 for prefill;
/// KV-cache length for single-token decode).
class Transformer {
public:
    struct Weights {
        Tensor<float> attn_norm;  // [hidden]
        Tensor<float> w_q;        // [hidden, q_dim]
        Tensor<float> w_k;        // [hidden, kv_dim]
        Tensor<float> w_v;        // [hidden, kv_dim]
        Tensor<float> w_o;        // [q_dim, hidden]
        Tensor<float> ffn_norm;   // [hidden]
        Tensor<float> w_gate;     // [hidden, intermediate]
        Tensor<float> w_up;       // [hidden, intermediate]
        Tensor<float> w_down;     // [intermediate, hidden]
    };

    /// Allocate zero-filled weights with shapes implied by `config`.
    explicit Transformer(TransformerConfig config);

    /// Take ownership of already-allocated weights. Shapes must match `config`.
    Transformer(TransformerConfig config, Weights weights);

    const TransformerConfig& config() const { return config_; }
    Weights& weights() { return weights_; }
    const Weights& weights() const { return weights_; }

    void forward(const Tensor<float>& x, Tensor<float>& out, size_t position_offset = 0) const;
    Tensor<float> forward(const Tensor<float>& x, size_t position_offset = 0) const;

private:
    void forward_seq(const Tensor<float>& x, Tensor<float>& out, size_t position_offset) const;

    TransformerConfig config_;
    Weights weights_;
};

}  // namespace llmberry
