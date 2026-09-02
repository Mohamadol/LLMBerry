#include "llmberry/attention.h"
#include "llmberry/kernels.h"
#include "llmberry/mlp.h"
#include "llmberry/tensor.h"
#include "llmberry/transformer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using llmberry::add;
using llmberry::attention;
using llmberry::matmul;
using llmberry::mlp;
using llmberry::rmsnorm;
using llmberry::rope;
using llmberry::Tensor;
using llmberry::Transformer;
using llmberry::TransformerConfig;

constexpr float kTol = 1e-4f;

void fill_iota(Tensor<float>& t, float start = 1.0f) {
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = start + static_cast<float>(i);
    }
}

void fill_pattern(Tensor<float>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
    }
}

void fill_identity(Tensor<float>& t) {
    ASSERT_EQ(t.ndim(), 2u);
    ASSERT_EQ(t.shape()[0], t.shape()[1]);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = 0.0f;
    }
    for (size_t i = 0; i < t.shape()[0]; ++i) {
        t.at({i, i}) = 1.0f;
    }
}

void copy_tensor(const Tensor<float>& src, Tensor<float>& dst) {
    ASSERT_EQ(src.size(), dst.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i];
    }
}

void expect_allclose(const Tensor<float>& got, const Tensor<float>& want, float tol = kTol) {
    ASSERT_EQ(got.shape(), want.shape());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
    }
}

TransformerConfig tiny_config(size_t n_heads = 2, size_t n_kv_heads = 2, size_t head_dim = 2,
                              size_t intermediate = 6) {
    TransformerConfig c;
    c.hidden = n_heads * head_dim;
    c.n_heads = n_heads;
    c.n_kv_heads = n_kv_heads;
    c.intermediate = intermediate;
    return c;
}

void fill_uniform(Tensor<float>& t, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = dist(rng);
    }
}

void fill_weights_small(Transformer& block, std::mt19937& rng) {
    auto& w = block.weights();
    fill_uniform(w.attn_norm, rng, 0.5f, 1.5f);
    fill_uniform(w.w_q, rng, -0.2f, 0.2f);
    fill_uniform(w.w_k, rng, -0.2f, 0.2f);
    fill_uniform(w.w_v, rng, -0.2f, 0.2f);
    fill_uniform(w.w_o, rng, -0.2f, 0.2f);
    fill_uniform(w.ffn_norm, rng, 0.5f, 1.5f);
    fill_uniform(w.w_gate, rng, -0.2f, 0.2f);
    fill_uniform(w.w_up, rng, -0.2f, 0.2f);
    fill_uniform(w.w_down, rng, -0.2f, 0.2f);
}

/// Llama 3 8B GQA 32:8 scaled down: head_dim=64, n_kv = n_heads/4, MLP ≈ 3.5×H.
TransformerConfig mini_llama3_gqa() {
    TransformerConfig c;
    c.hidden = 256;
    c.n_heads = 4;
    c.n_kv_heads = 1;
    c.intermediate = 896;
    return c;
}

/// Llama 2 7B MHA scaled down: head_dim=64, MLP ≈ 2.69×H (11008/4096).
TransformerConfig mini_llama2_mha() {
    TransformerConfig c;
    c.hidden = 256;
    c.n_heads = 4;
    c.n_kv_heads = 4;
    c.intermediate = 688;
    return c;
}

void fill_weights(Transformer& block) {
    auto& w = block.weights();
    fill_pattern(w.attn_norm);
    for (size_t i = 0; i < w.attn_norm.size(); ++i) {
        w.attn_norm[i] += 1.0f;
    }
    fill_iota(w.w_q, 0.01f);
    fill_pattern(w.w_k);
    fill_iota(w.w_v, -0.02f);
    fill_pattern(w.w_o);
    fill_iota(w.ffn_norm, 0.5f);
    fill_pattern(w.w_gate);
    fill_iota(w.w_up, -0.05f);
    fill_pattern(w.w_down);
}

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

/// Independent glue: same pipeline as Transformer::forward, written in the test.
void transformer_ref(const Transformer& block, const Tensor<float>& x, Tensor<float>& out,
                     size_t position_offset = 0) {
    const auto& c = block.config();
    const auto& w = block.weights();
    Tensor<float> x2d = (x.ndim() == 1) ? x.view({1, c.hidden}) : Tensor<float>();
    Tensor<float> out2d = (x.ndim() == 1) ? out.view({1, c.hidden}) : Tensor<float>();
    const Tensor<float>& xin = (x.ndim() == 1) ? x2d : x;
    Tensor<float>& o = (x.ndim() == 1) ? out2d : out;

    if (x.ndim() == 3) {
        for (size_t b = 0; b < x.shape()[0]; ++b) {
            Tensor<float> xb = x.matrix(b);
            Tensor<float> ob = out.matrix(b);
            Tensor<float> want(xb.shape());
            transformer_ref(block, xb, want, position_offset);
            copy_tensor(want, ob);
        }
        return;
    }

    const size_t seq = xin.shape()[0];
    const size_t n_q = c.n_heads;
    const size_t n_kv = c.n_kv_heads;
    const size_t d = c.head_dim();

    Tensor<float> xn = rmsnorm(xin, w.attn_norm, c.rms_eps);
    Tensor<float> q({seq, n_q * d});
    Tensor<float> k({seq, n_kv * d});
    Tensor<float> v({seq, n_kv * d});
    matmul(xn, w.w_q, q);
    matmul(xn, w.w_k, k);
    matmul(xn, w.w_v, v);

    Tensor<float> q_rot = rope(q.view({seq, n_q, d}), position_offset, c.rope_theta);
    Tensor<float> k_rot = rope(k.view({seq, n_kv, d}), position_offset, c.rope_theta);
    Tensor<float> q_hs({n_q, seq, d});
    Tensor<float> k_hs({n_kv, seq, d});
    Tensor<float> v_hs({n_kv, seq, d});
    seq_heads_to_heads_seq(q_rot, q_hs);
    seq_heads_to_heads_seq(k_rot, k_hs);
    seq_heads_to_heads_seq(v.view({seq, n_kv, d}), v_hs);

    Tensor<float> attn_hs = attention(q_hs, k_hs, v_hs, true);
    Tensor<float> attn_sh({seq, n_q, d});
    heads_seq_to_seq_heads(attn_hs, attn_sh);
    Tensor<float> attn_out({seq, c.hidden});
    matmul(attn_sh.view({seq, n_q * d}), w.w_o, attn_out);

    Tensor<float> h = add(xin, attn_out);
    Tensor<float> mlp_out = mlp(rmsnorm(h, w.ffn_norm, c.rms_eps), w.w_gate, w.w_up, w.w_down);
    add(h, mlp_out, o);
}

Tensor<float> prefix_rows(const Tensor<float>& x, size_t n) {
    Tensor<float> p({n, x.shape()[1]});
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < x.shape()[1]; ++j) {
            p.at({t, j}) = x.at({t, j});
        }
    }
    return p;
}

}  // namespace

TEST(Transformer, AllocatesWeightShapes) {
    TransformerConfig c = tiny_config();
    Transformer block(c);
    EXPECT_EQ(block.config().hidden, 4u);
    EXPECT_EQ(block.config().head_dim(), 2u);
    EXPECT_EQ(block.weights().attn_norm.shape(), (std::vector<size_t>{4}));
    EXPECT_EQ(block.weights().w_q.shape(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(block.weights().w_k.shape(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(block.weights().w_v.shape(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(block.weights().w_o.shape(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(block.weights().ffn_norm.shape(), (std::vector<size_t>{4}));
    EXPECT_EQ(block.weights().w_gate.shape(), (std::vector<size_t>{4, 6}));
    EXPECT_EQ(block.weights().w_up.shape(), (std::vector<size_t>{4, 6}));
    EXPECT_EQ(block.weights().w_down.shape(), (std::vector<size_t>{6, 4}));
}

TEST(Transformer, GqaAllocatesSmallerKv) {
    Transformer block(tiny_config(/*n_heads=*/4, /*n_kv_heads=*/2, /*head_dim=*/2));
    EXPECT_EQ(block.weights().w_q.shape(), (std::vector<size_t>{8, 8}));
    EXPECT_EQ(block.weights().w_k.shape(), (std::vector<size_t>{8, 4}));
    EXPECT_EQ(block.weights().w_v.shape(), (std::vector<size_t>{8, 4}));
    EXPECT_EQ(block.weights().w_o.shape(), (std::vector<size_t>{8, 8}));
}

TEST(Transformer, InvalidConfigThrows) {
    TransformerConfig c = tiny_config();
    c.n_heads = 3;
    EXPECT_THROW((Transformer(c)), std::invalid_argument);

    c = tiny_config(/*n_heads=*/4, /*n_kv_heads=*/3, /*head_dim=*/2);
    c.hidden = 8;
    EXPECT_THROW((Transformer(c)), std::invalid_argument);

    c = tiny_config();
    c.n_heads = 0;
    EXPECT_THROW((Transformer(c)), std::invalid_argument);

    c = tiny_config(/*n_heads=*/2, /*n_kv_heads=*/2, /*head_dim=*/3);
    c.hidden = 6;
    EXPECT_THROW((Transformer(c)), std::invalid_argument);
}

TEST(Transformer, WeightShapeMismatchThrows) {
    TransformerConfig c = tiny_config();
    Transformer::Weights w;
    w.attn_norm = Tensor<float>::zeros({4});
    w.w_q = Tensor<float>::zeros({4, 4});
    w.w_k = Tensor<float>::zeros({4, 4});
    w.w_v = Tensor<float>::zeros({4, 4});
    w.w_o = Tensor<float>::zeros({4, 3});
    w.ffn_norm = Tensor<float>::zeros({4});
    w.w_gate = Tensor<float>::zeros({4, 6});
    w.w_up = Tensor<float>::zeros({4, 6});
    w.w_down = Tensor<float>::zeros({6, 4});
    EXPECT_THROW((Transformer(c, std::move(w))), std::invalid_argument);
}

TEST(Transformer, ZeroProjectionsIsIdentity) {
    Transformer block(tiny_config());
    fill_weights(block);
    for (size_t i = 0; i < block.weights().w_o.size(); ++i) {
        block.weights().w_o[i] = 0.0f;
    }
    for (size_t i = 0; i < block.weights().w_down.size(); ++i) {
        block.weights().w_down[i] = 0.0f;
    }

    Tensor<float> x({3, 4});
    fill_iota(x, -2.0f);
    Tensor<float> out({3, 4});
    fill_iota(out, 99.0f);
    block.forward(x, out);
    expect_allclose(out, x);
}

TEST(Transformer, KnownValuesTiny) {
    TransformerConfig c;
    c.hidden = 2;
    c.n_heads = 1;
    c.n_kv_heads = 1;
    c.intermediate = 2;
    Transformer block(c);
    block.weights().attn_norm[0] = 1.0f;
    block.weights().attn_norm[1] = 1.0f;
    block.weights().ffn_norm[0] = 1.0f;
    block.weights().ffn_norm[1] = 1.0f;
    fill_identity(block.weights().w_q);
    fill_identity(block.weights().w_k);
    fill_identity(block.weights().w_v);
    fill_identity(block.weights().w_o);

    Tensor<float> x({2});
    x[0] = 1.0f;
    x[1] = 0.0f;
    Tensor<float> out = block.forward(x);

    const float inv_rms = 1.0f / std::sqrt(0.5f + 1e-6f);
    EXPECT_NEAR(out[0], 1.0f + inv_rms, kTol);
    EXPECT_NEAR(out[1], 0.0f, kTol);
}

TEST(Transformer, PrefillLastRowEqualsPrefix) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> x({4, 4});
    fill_iota(x, -1.5f);
    Tensor<float> full = block.forward(x);

    for (size_t t = 0; t < 4; ++t) {
        Tensor<float> pref = prefix_rows(x, t + 1);
        Tensor<float> y = block.forward(pref);
        Tensor<float> last({4});
        copy_tensor(y.row(t), last);
        Tensor<float> want({4});
        copy_tensor(full.row(t), want);
        expect_allclose(last, want);
    }
}

TEST(Transformer, DecodeMatchesPrefillSeq1) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> x({4});
    fill_pattern(x);
    Tensor<float> out_1d = block.forward(x);
    Tensor<float> x2d = x.view({1, 4});
    Tensor<float> out_2d = block.forward(x2d);
    expect_allclose(out_1d, out_2d.view({4}));
}

TEST(Transformer, MatchesIndependentReference) {
    {
        Transformer block(tiny_config());
        fill_weights(block);
        Tensor<float> x({4});
        fill_iota(x, -2.0f);
        Tensor<float> out({4});
        Tensor<float> want({4});
        block.forward(x, out);
        transformer_ref(block, x, want);
        expect_allclose(out, want);
    }
    {
        Transformer block(tiny_config());
        fill_weights(block);
        Tensor<float> x({3, 4});
        fill_iota(x, 1.0f);
        Tensor<float> out({3, 4});
        Tensor<float> want({3, 4});
        block.forward(x, out);
        transformer_ref(block, x, want);
        expect_allclose(out, want);
    }
    {
        Transformer block(tiny_config(/*n_heads=*/4, /*n_kv_heads=*/2, /*head_dim=*/2, /*intermediate=*/5));
        fill_weights(block);
        Tensor<float> x({2, 3, 8});
        fill_pattern(x);
        Tensor<float> out({2, 3, 8});
        Tensor<float> want({2, 3, 8});
        block.forward(x, out);
        transformer_ref(block, x, want);
        expect_allclose(out, want);
    }
}

TEST(Transformer, RealisticMiniLlamaGqaMatchesRef) {
    std::mt19937 rng(1);
    Transformer block(mini_llama3_gqa());
    fill_weights_small(block, rng);
    EXPECT_EQ(block.config().head_dim(), 64u);

    Tensor<float> x({8, 256});
    fill_uniform(x, rng, -1.0f, 1.0f);
    Tensor<float> out({8, 256});
    Tensor<float> want({8, 256});
    block.forward(x, out);
    transformer_ref(block, x, want);
    expect_allclose(out, want);

    Tensor<float> tok({256});
    fill_uniform(tok, rng, -1.0f, 1.0f);
    Tensor<float> tok_out({256});
    Tensor<float> tok_want({256});
    block.forward(tok, tok_out, /*position_offset=*/17);
    transformer_ref(block, tok, tok_want, 17);
    expect_allclose(tok_out, tok_want);
}

TEST(Transformer, RealisticMiniLlama2MhaMatchesRef) {
    std::mt19937 rng(2);
    Transformer block(mini_llama2_mha());
    fill_weights_small(block, rng);
    EXPECT_EQ(block.config().head_dim(), 64u);
    EXPECT_EQ(block.config().n_heads, block.config().n_kv_heads);

    Tensor<float> x({16, 256});
    fill_uniform(x, rng, -1.0f, 1.0f);
    Tensor<float> out({16, 256});
    Tensor<float> want({16, 256});
    block.forward(x, out);
    transformer_ref(block, x, want);
    expect_allclose(out, want);
}

TEST(Transformer, RealisticHeadDim128MatchesRef) {
    std::mt19937 rng(3);
    TransformerConfig c;
    c.hidden = 256;
    c.n_heads = 2;
    c.n_kv_heads = 2;
    c.intermediate = 688;
    Transformer block(c);
    fill_weights_small(block, rng);
    EXPECT_EQ(block.config().head_dim(), 128u);

    Tensor<float> x({8, 256});
    fill_uniform(x, rng, -1.0f, 1.0f);
    Tensor<float> out({8, 256});
    Tensor<float> want({8, 256});
    block.forward(x, out);
    transformer_ref(block, x, want);
    expect_allclose(out, want);
}

TEST(Transformer, RealisticGqaPrefillLastRowEqualsPrefix) {
    std::mt19937 rng(4);
    Transformer block(mini_llama3_gqa());
    fill_weights_small(block, rng);
    Tensor<float> x({4, 256});
    fill_uniform(x, rng, -1.0f, 1.0f);
    Tensor<float> full = block.forward(x);

    for (size_t t = 0; t < 4; ++t) {
        Tensor<float> pref = prefix_rows(x, t + 1);
        Tensor<float> y = block.forward(pref);
        Tensor<float> last({256});
        copy_tensor(y.row(t), last);
        Tensor<float> want({256});
        copy_tensor(full.row(t), want);
        expect_allclose(last, want);
    }
}

TEST(Transformer, OverwritesOut) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> x({2, 4});
    Tensor<float> out({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(x);
    fill_iota(out, 100.0f);
    block.forward(x, out);
    transformer_ref(block, x, want);
    expect_allclose(out, want);
}

TEST(Transformer, LeavesInputsUnchanged) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> x({2, 4});
    fill_pattern(x);
    Tensor<float> x_copy({2, 4});
    copy_tensor(x, x_copy);
    auto snapshot = [&](const Tensor<float>& src) {
        Tensor<float> dst(src.shape());
        copy_tensor(src, dst);
        return dst;
    };
    Tensor<float> attn_norm = snapshot(block.weights().attn_norm);
    Tensor<float> w_q = snapshot(block.weights().w_q);
    Tensor<float> w_k = snapshot(block.weights().w_k);
    Tensor<float> w_v = snapshot(block.weights().w_v);
    Tensor<float> w_o = snapshot(block.weights().w_o);
    Tensor<float> ffn_norm = snapshot(block.weights().ffn_norm);
    Tensor<float> w_gate = snapshot(block.weights().w_gate);
    Tensor<float> w_up = snapshot(block.weights().w_up);
    Tensor<float> w_down = snapshot(block.weights().w_down);

    Tensor<float> out({2, 4});
    block.forward(x, out);

    expect_allclose(x, x_copy);
    expect_allclose(block.weights().attn_norm, attn_norm);
    expect_allclose(block.weights().w_q, w_q);
    expect_allclose(block.weights().w_k, w_k);
    expect_allclose(block.weights().w_v, w_v);
    expect_allclose(block.weights().w_o, w_o);
    expect_allclose(block.weights().ffn_norm, ffn_norm);
    expect_allclose(block.weights().w_gate, w_gate);
    expect_allclose(block.weights().w_up, w_up);
    expect_allclose(block.weights().w_down, w_down);
}

TEST(Transformer, AllocatingWrapper) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> x({2, 4});
    fill_iota(x, -1.0f);
    Tensor<float> want({2, 4});
    transformer_ref(block, x, want);
    Tensor<float> out = block.forward(x);
    EXPECT_EQ(out.shape(), x.shape());
    expect_allclose(out, want);
}

TEST(Transformer, ContiguousReshapeView) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> storage({24});
    fill_iota(storage, 1.0f);
    Tensor<float> x = storage.view({2, 3, 4});
    Tensor<float> out({2, 3, 4});
    Tensor<float> want({2, 3, 4});
    block.forward(x, out);
    transformer_ref(block, x, want);
    expect_allclose(out, want);
}

TEST(Transformer, ShapeMismatchThrows) {
    Transformer block(tiny_config());
    fill_weights(block);
    Tensor<float> x({2, 4});
    Tensor<float> out({2, 4});
    Tensor<float> out_bad({4, 2});
    Tensor<float> x_hidden({2, 5});
    Tensor<float> out_hidden({2, 5});
    EXPECT_THROW(block.forward(x, out_bad), std::invalid_argument);
    EXPECT_THROW(block.forward(x_hidden, out_hidden), std::invalid_argument);
}

TEST(Transformer, EmptyThrows) {
    Transformer block(tiny_config());
    Tensor<float> x;
    Tensor<float> out;
    EXPECT_THROW(block.forward(x, out), std::invalid_argument);
    EXPECT_THROW(block.forward(x), std::invalid_argument);
}
