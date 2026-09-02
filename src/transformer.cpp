#include "llmberry/transformer.h"
#include "llmberry/attention.h"
#include "llmberry/ensure.h"
#include "llmberry/kernels.h"
#include "llmberry/mlp.h"

#include <stdexcept>
#include <utility>

namespace llmberry {
namespace {

void validate_config(const TransformerConfig& c) {
    ENSURE(c.hidden > 0, std::invalid_argument, "Transformer: hidden must be positive");
    ENSURE(c.n_heads > 0, std::invalid_argument, "Transformer: n_heads must be positive");
    ENSURE(c.n_kv_heads > 0, std::invalid_argument, "Transformer: n_kv_heads must be positive");
    ENSURE(c.intermediate > 0, std::invalid_argument, "Transformer: intermediate must be positive");
    ENSURE(c.hidden % c.n_heads == 0, std::invalid_argument,
           "Transformer: hidden must be divisible by n_heads");
    ENSURE(c.n_heads % c.n_kv_heads == 0, std::invalid_argument,
           "Transformer: n_heads must be divisible by n_kv_heads");
    ENSURE(c.head_dim() % 2 == 0, std::invalid_argument,
           "Transformer: head_dim must be even (RoPE)");
    ENSURE(c.rope_theta > 0.0f, std::invalid_argument, "Transformer: rope_theta must be positive");
    ENSURE(c.rms_eps > 0.0f, std::invalid_argument, "Transformer: rms_eps must be positive");
}

void validate_weights(const TransformerConfig& c, const Transformer::Weights& w) {
    const size_t h = c.hidden;
    const size_t qd = c.q_dim();
    const size_t kvd = c.kv_dim();
    const size_t inter = c.intermediate;

    ENSURE(!w.attn_norm.empty() && w.attn_norm.ndim() == 1 && w.attn_norm.size() == h,
           std::invalid_argument, "Transformer: attn_norm must have shape [hidden]");
    ENSURE(w.w_q.ndim() == 2 && w.w_q.shape()[0] == h && w.w_q.shape()[1] == qd,
           std::invalid_argument, "Transformer: w_q must have shape [hidden, n_heads * head_dim]");
    ENSURE(w.w_k.ndim() == 2 && w.w_k.shape()[0] == h && w.w_k.shape()[1] == kvd,
           std::invalid_argument, "Transformer: w_k must have shape [hidden, n_kv_heads * head_dim]");
    ENSURE(w.w_v.ndim() == 2 && w.w_v.shape()[0] == h && w.w_v.shape()[1] == kvd,
           std::invalid_argument, "Transformer: w_v must have shape [hidden, n_kv_heads * head_dim]");
    ENSURE(w.w_o.ndim() == 2 && w.w_o.shape()[0] == qd && w.w_o.shape()[1] == h,
           std::invalid_argument, "Transformer: w_o must have shape [n_heads * head_dim, hidden]");
    ENSURE(!w.ffn_norm.empty() && w.ffn_norm.ndim() == 1 && w.ffn_norm.size() == h,
           std::invalid_argument, "Transformer: ffn_norm must have shape [hidden]");
    ENSURE(w.w_gate.ndim() == 2 && w.w_gate.shape()[0] == h && w.w_gate.shape()[1] == inter,
           std::invalid_argument, "Transformer: w_gate must have shape [hidden, intermediate]");
    ENSURE(w.w_up.ndim() == 2 && w.w_up.shape()[0] == h && w.w_up.shape()[1] == inter,
           std::invalid_argument, "Transformer: w_up must have shape [hidden, intermediate]");
    ENSURE(w.w_down.ndim() == 2 && w.w_down.shape()[0] == inter && w.w_down.shape()[1] == h,
           std::invalid_argument, "Transformer: w_down must have shape [intermediate, hidden]");
}

Transformer::Weights allocate_weights(const TransformerConfig& c) {
    Transformer::Weights w;
    w.attn_norm = Tensor<float>::zeros({c.hidden});
    w.w_q = Tensor<float>::zeros({c.hidden, c.q_dim()});
    w.w_k = Tensor<float>::zeros({c.hidden, c.kv_dim()});
    w.w_v = Tensor<float>::zeros({c.hidden, c.kv_dim()});
    w.w_o = Tensor<float>::zeros({c.q_dim(), c.hidden});
    w.ffn_norm = Tensor<float>::zeros({c.hidden});
    w.w_gate = Tensor<float>::zeros({c.hidden, c.intermediate});
    w.w_up = Tensor<float>::zeros({c.hidden, c.intermediate});
    w.w_down = Tensor<float>::zeros({c.intermediate, c.hidden});
    return w;
}

/// src [seq, n_h, d] -> dst [n_h, seq, d]
void seq_heads_to_heads_seq(const Tensor<float>& src, Tensor<float>& dst) {
    const size_t seq = src.shape()[0];
    const size_t n_h = src.shape()[1];
    const size_t d = src.shape()[2];
    for (size_t t = 0; t < seq; ++t) {
        for (size_t h = 0; h < n_h; ++h) {
            for (size_t i = 0; i < d; ++i) {
                dst.at({h, t, i}) = src.at({t, h, i});
            }
        }
    }
}

/// src [n_h, seq, d] -> dst [seq, n_h, d]
void heads_seq_to_seq_heads(const Tensor<float>& src, Tensor<float>& dst) {
    const size_t n_h = src.shape()[0];
    const size_t seq = src.shape()[1];
    const size_t d = src.shape()[2];
    for (size_t h = 0; h < n_h; ++h) {
        for (size_t t = 0; t < seq; ++t) {
            for (size_t i = 0; i < d; ++i) {
                dst.at({t, h, i}) = src.at({h, t, i});
            }
        }
    }
}

}  // namespace

Transformer::Transformer(TransformerConfig config) : config_(config) {
    validate_config(config_);
    weights_ = allocate_weights(config_);
}

Transformer::Transformer(TransformerConfig config, Weights weights)
    : config_(config), weights_(std::move(weights)) {
    validate_config(config_);
    validate_weights(config_, weights_);
}

void Transformer::forward(const Tensor<float>& x, Tensor<float>& out, size_t position_offset) const {
    ENSURE(!x.empty(), std::invalid_argument, "Transformer: x cannot be empty");
    ENSURE(x.ndim() >= 1 && x.ndim() <= 3, std::invalid_argument,
           "Transformer: x must have shape [hidden], [seq, hidden], or [batch, seq, hidden]");
    ENSURE(x.shape().back() == config_.hidden, std::invalid_argument,
           "Transformer: x last dim must equal hidden");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "Transformer: out shape must match x");

    if (x.ndim() == 3) {
        const size_t batch = x.shape()[0];
        for (size_t b = 0; b < batch; ++b) {
            Tensor<float> x_b = x.matrix(b);
            Tensor<float> out_b = out.matrix(b);
            forward_seq(x_b, out_b, position_offset);
        }
        return;
    }
    if (x.ndim() == 1) {
        Tensor<float> x2d = x.view({1, config_.hidden});
        Tensor<float> out2d = out.view({1, config_.hidden});
        forward_seq(x2d, out2d, position_offset);
        return;
    }
    forward_seq(x, out, position_offset);
}

Tensor<float> Transformer::forward(const Tensor<float>& x, size_t position_offset) const {
    ENSURE(!x.empty(), std::invalid_argument, "Transformer: x cannot be empty");
    Tensor<float> out(x.shape());
    forward(x, out, position_offset);
    return out;
}

void Transformer::forward_seq(const Tensor<float>& x, Tensor<float>& out, size_t position_offset) const {
    const size_t seq = x.shape()[0];
    const size_t hidden = config_.hidden;
    const size_t n_q = config_.n_heads;
    const size_t n_kv = config_.n_kv_heads;
    const size_t d = config_.head_dim();

    Tensor<float> xn(x.shape());
    rmsnorm(x, weights_.attn_norm, xn, config_.rms_eps);

    Tensor<float> q({seq, n_q * d});
    Tensor<float> k({seq, n_kv * d});
    Tensor<float> v({seq, n_kv * d});
    matmul(xn, weights_.w_q, q);
    matmul(xn, weights_.w_k, k);
    matmul(xn, weights_.w_v, v);

    Tensor<float> q_sh = q.view({seq, n_q, d});
    Tensor<float> k_sh = k.view({seq, n_kv, d});
    Tensor<float> v_sh = v.view({seq, n_kv, d});

    Tensor<float> q_rot(q_sh.shape());
    Tensor<float> k_rot(k_sh.shape());
    rope(q_sh, q_rot, position_offset, config_.rope_theta);
    rope(k_sh, k_rot, position_offset, config_.rope_theta);

    Tensor<float> q_hs({n_q, seq, d});
    Tensor<float> k_hs({n_kv, seq, d});
    Tensor<float> v_hs({n_kv, seq, d});
    seq_heads_to_heads_seq(q_rot, q_hs);
    seq_heads_to_heads_seq(k_rot, k_hs);
    seq_heads_to_heads_seq(v_sh, v_hs);

    Tensor<float> attn_hs({n_q, seq, d});
    attention(q_hs, k_hs, v_hs, attn_hs, true);

    Tensor<float> attn_sh({seq, n_q, d});
    heads_seq_to_seq_heads(attn_hs, attn_sh);
    Tensor<float> attn_2d = attn_sh.view({seq, n_q * d});

    Tensor<float> attn_out({seq, hidden});
    matmul(attn_2d, weights_.w_o, attn_out);

    Tensor<float> h(x.shape());
    add(x, attn_out, h);

    Tensor<float> hn(x.shape());
    rmsnorm(h, weights_.ffn_norm, hn, config_.rms_eps);

    Tensor<float> mlp_out(x.shape());
    mlp(hn, weights_.w_gate, weights_.w_up, weights_.w_down, mlp_out);

    add(h, mlp_out, out);
}

}  // namespace llmberry
