#include "llmberry/attention.h"
#include "llmberry/kernels.h"
#include "llmberry/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using llmberry::attention;
using llmberry::attention_av;
using llmberry::attention_causal_mask;
using llmberry::attention_qk;
using llmberry::attention_scale;
using llmberry::softmax;
using llmberry::Tensor;

constexpr float kTol = 1e-5f;
const float kNegInf = -std::numeric_limits<float>::infinity();

void fill_iota(Tensor<float>& t, float start = 1.0f) {
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = start + static_cast<float>(i);
    }
}

void fill_pattern(Tensor<float>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
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
        if (std::isinf(want[i]) && std::isinf(got[i]) && (want[i] < 0.0f) == (got[i] < 0.0f)) {
            continue;
        }
        EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
    }
}

void softmax_ref(const Tensor<float>& x, Tensor<float>& out) {
    const size_t dim = x.shape().back();
    const size_t n_rows = x.size() / dim;
    for (size_t r = 0; r < n_rows; ++r) {
        float m = x[r * dim];
        for (size_t i = 1; i < dim; ++i) {
            m = std::max(m, x[r * dim + i]);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            const float e = std::exp(x[r * dim + i] - m);
            out[r * dim + i] = e;
            sum += e;
        }
        for (size_t i = 0; i < dim; ++i) {
            out[r * dim + i] /= sum;
        }
    }
}

void attention_qk_ref(const Tensor<float>& q, const Tensor<float>& k, Tensor<float>& scores) {
    const size_t seq_q = q.shape()[q.ndim() - 2];
    const size_t d = q.shape().back();
    const size_t seq_k = k.shape()[k.ndim() - 2];
    const size_t n_heads = q.size() / (seq_q * d);
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t i = 0; i < seq_q; ++i) {
            for (size_t j = 0; j < seq_k; ++j) {
                float acc = 0.0f;
                for (size_t t = 0; t < d; ++t) {
                    acc += q[h * seq_q * d + i * d + t] * k[h * seq_k * d + j * d + t];
                }
                scores[h * seq_q * seq_k + i * seq_k + j] = acc;
            }
        }
    }
}

void attention_causal_mask_ref(Tensor<float>& scores) {
    const size_t seq_q = scores.shape()[scores.ndim() - 2];
    const size_t seq_k = scores.shape().back();
    const size_t n_heads = scores.size() / (seq_q * seq_k);
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t i = 0; i < seq_q; ++i) {
            for (size_t j = 0; j < seq_k; ++j) {
                if (j > i + seq_k - seq_q) {
                    scores[h * seq_q * seq_k + i * seq_k + j] = kNegInf;
                }
            }
        }
    }
}

void attention_av_ref(const Tensor<float>& attn, const Tensor<float>& v, Tensor<float>& out) {
    const size_t seq_q = attn.shape()[attn.ndim() - 2];
    const size_t seq_k = attn.shape().back();
    const size_t d_v = v.shape().back();
    const size_t n_heads = v.size() / (seq_k * d_v);
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t i = 0; i < seq_q; ++i) {
            for (size_t t = 0; t < d_v; ++t) {
                float acc = 0.0f;
                for (size_t j = 0; j < seq_k; ++j) {
                    acc += attn[h * seq_q * seq_k + i * seq_k + j] *
                           v[h * seq_k * d_v + j * d_v + t];
                }
                out[h * seq_q * d_v + i * d_v + t] = acc;
            }
        }
    }
}

void attention_ref(const Tensor<float>& q,
                   const Tensor<float>& k,
                   const Tensor<float>& v,
                   Tensor<float>& out,
                   bool causal = true,
                   float scale = -1.0f) {
    const size_t d = q.shape().back();
    const float s = (scale < 0.0f) ? 1.0f / std::sqrt(static_cast<float>(d)) : scale;
    std::vector<size_t> sc_shape = q.shape();
    sc_shape.back() = k.shape()[k.ndim() - 2];
    Tensor<float> scores(sc_shape);
    Tensor<float> weights(sc_shape);
    attention_qk_ref(q, k, scores);
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] *= s;
    }
    if (causal) {
        attention_causal_mask_ref(scores);
    }
    softmax_ref(scores, weights);
    attention_av_ref(weights, v, out);
}

void copy_head(const Tensor<float>& src, size_t h, Tensor<float>& dst) {
    const size_t seq = src.shape()[1];
    const size_t dim = src.shape()[2];
    ASSERT_EQ(dst.shape(), (std::vector<size_t>{seq, dim}));
    for (size_t i = 0; i < seq; ++i) {
        for (size_t t = 0; t < dim; ++t) {
            dst.at({i, t}) = src.at({h, i, t});
        }
    }
}

size_t n_heads_of(const Tensor<float>& t) {
    return (t.ndim() >= 3) ? t.shape()[t.ndim() - 3] : 1;
}

/// Repeat-interleave KV heads along the head axis so GQA becomes MHA.
/// Rank 3: [n_kv, seq, d] -> [n_q, seq, d]
/// Rank 4: [B, n_kv, seq, d] -> [B, n_q, seq, d]
Tensor<float> expand_kv_heads(const Tensor<float>& kv, size_t n_q) {
    if (kv.ndim() < 3) {
        Tensor<float> out(kv.shape());
        copy_tensor(kv, out);
        return out;
    }

    std::vector<size_t> shape = kv.shape();
    const size_t n_kv = shape[kv.ndim() - 3];
    const size_t group = n_q / n_kv;
    shape[kv.ndim() - 3] = n_q;
    Tensor<float> out(shape);

    const size_t seq = kv.shape()[kv.ndim() - 2];
    const size_t d = kv.shape().back();

    if (kv.ndim() == 3) {
        for (size_t kv_h = 0; kv_h < n_kv; ++kv_h) {
            for (size_t g = 0; g < group; ++g) {
                const size_t q_h = kv_h * group + g;
                for (size_t i = 0; i < seq; ++i) {
                    for (size_t t = 0; t < d; ++t) {
                        out.at({q_h, i, t}) = kv.at({kv_h, i, t});
                    }
                }
            }
        }
        return out;
    }

    if (kv.ndim() != 4) {
        ADD_FAILURE() << "expand_kv_heads: expected rank 3 or 4, got " << kv.ndim();
        return out;
    }
    const size_t batch = kv.shape()[0];
    for (size_t b = 0; b < batch; ++b) {
        for (size_t kv_h = 0; kv_h < n_kv; ++kv_h) {
            for (size_t g = 0; g < group; ++g) {
                const size_t q_h = kv_h * group + g;
                for (size_t i = 0; i < seq; ++i) {
                    for (size_t t = 0; t < d; ++t) {
                        out.at({b, q_h, i, t}) = kv.at({b, kv_h, i, t});
                    }
                }
            }
        }
    }
    return out;
}

void attention_gqa_ref(const Tensor<float>& q,
                       const Tensor<float>& k,
                       const Tensor<float>& v,
                       Tensor<float>& out,
                       bool causal = true,
                       float scale = -1.0f) {
    const size_t n_q = n_heads_of(q);
    Tensor<float> k_exp = expand_kv_heads(k, n_q);
    Tensor<float> v_exp = expand_kv_heads(v, n_q);
    attention_ref(q, k_exp, v_exp, out, causal, scale);
}

void copy_seq_prefix(const Tensor<float>& src, Tensor<float>& dst) {
    const size_t seq_dst = dst.shape()[dst.ndim() - 2];
    const size_t d = dst.shape().back();
    ASSERT_EQ(src.shape().back(), d);
    ASSERT_GE(src.shape()[src.ndim() - 2], seq_dst);
    const size_t n = dst.size() / (seq_dst * d);
    ASSERT_EQ(n, src.size() / (src.shape()[src.ndim() - 2] * d));
    for (size_t h = 0; h < n; ++h) {
        Tensor<float> s = src.matrix(h);
        Tensor<float> o = dst.matrix(h);
        for (size_t i = 0; i < seq_dst; ++i) {
            for (size_t t = 0; t < d; ++t) {
                o.at({i, t}) = s.at({i, t});
            }
        }
    }
}

void copy_query_at(const Tensor<float>& src, size_t pos, Tensor<float>& dst) {
    ASSERT_EQ(dst.shape()[dst.ndim() - 2], 1u);
    const size_t d = src.shape().back();
    ASSERT_EQ(dst.shape().back(), d);
    const size_t n = dst.size() / d;
    ASSERT_EQ(n, src.size() / (src.shape()[src.ndim() - 2] * d));
    for (size_t h = 0; h < n; ++h) {
        Tensor<float> s = src.matrix(h);
        Tensor<float> o = dst.matrix(h);
        for (size_t t = 0; t < d; ++t) {
            o.at({0, t}) = s.at({pos, t});
        }
    }
}

std::vector<size_t> with_seq(std::vector<size_t> shape, size_t seq) {
    shape[shape.size() - 2] = seq;
    return shape;
}

}  // namespace

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

TEST(Softmax, KnownValues1D) {
    Tensor<float> x({3});
    Tensor<float> out({3});
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;

    softmax(x, out);

    const float e0 = std::exp(-2.0f);
    const float e1 = std::exp(-1.0f);
    const float e2 = 1.0f;
    const float sum = e0 + e1 + e2;
    EXPECT_NEAR(out[0], e0 / sum, kTol);
    EXPECT_NEAR(out[1], e1 / sum, kTol);
    EXPECT_NEAR(out[2], e2 / sum, kTol);
}

TEST(Softmax, SumsToOne) {
    Tensor<float> x({5});
    Tensor<float> out({5});
    fill_pattern(x);
    softmax(x, out);

    float sum = 0.0f;
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_GE(out[i], 0.0f);
        sum += out[i];
    }
    EXPECT_NEAR(sum, 1.0f, kTol);
}

TEST(Softmax, UniformWhenEqual) {
    Tensor<float> x({4});
    Tensor<float> out({4});
    for (size_t i = 0; i < 4; ++i) {
        x[i] = 2.5f;
    }
    softmax(x, out);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], 0.25f, kTol);
    }
}

TEST(Softmax, NumericallyStableLargeValues) {
    Tensor<float> small({3});
    Tensor<float> large({3});
    Tensor<float> out_small({3});
    Tensor<float> out_large({3});
    small[0] = 1.0f;
    small[1] = 2.0f;
    small[2] = 3.0f;
    large[0] = 1001.0f;
    large[1] = 1002.0f;
    large[2] = 1003.0f;

    softmax(small, out_small);
    softmax(large, out_large);
    expect_allclose(out_large, out_small);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(out_large[i])) << "index " << i;
    }
}

TEST(Softmax, NegInfBecomesZero) {
    Tensor<float> x({2});
    Tensor<float> out({2});
    x[0] = 0.0f;
    x[1] = kNegInf;

    softmax(x, out);

    EXPECT_NEAR(out[0], 1.0f, kTol);
    EXPECT_NEAR(out[1], 0.0f, kTol);
}

TEST(Softmax, LastDimIndependent) {
    Tensor<float> x({2, 3});
    Tensor<float> out({2, 3});
    fill_iota(x, -1.0f);
    softmax(x, out);

    Tensor<float> row0({3});
    Tensor<float> row1({3});
    Tensor<float> want0({3});
    Tensor<float> want1({3});
    for (size_t i = 0; i < 3; ++i) {
        row0[i] = x.at({0, i});
        row1[i] = x.at({1, i});
    }
    softmax_ref(row0, want0);
    softmax_ref(row1, want1);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(out.at({0, i}), want0[i], kTol);
        EXPECT_NEAR(out.at({1, i}), want1[i], kTol);
    }
}

TEST(Softmax, MatchesIndependentReference) {
    const std::vector<std::vector<size_t>> shapes = {{4}, {3, 5}, {2, 3, 8}, {2, 2, 3, 4}};
    for (const auto& shape : shapes) {
        Tensor<float> x(shape);
        Tensor<float> got(shape);
        Tensor<float> want(shape);
        fill_pattern(x);
        softmax(x, got);
        softmax_ref(x, want);
        expect_allclose(got, want);
    }
}

TEST(Softmax, OverwritesOutput) {
    Tensor<float> x({4});
    Tensor<float> out({4});
    Tensor<float> want({4});
    fill_iota(x, -2.0f);
    fill_iota(out, 100.0f);
    softmax(x, out);
    softmax_ref(x, want);
    expect_allclose(out, want);
}

TEST(Softmax, LeavesInputUnchanged) {
    Tensor<float> x({2, 4});
    Tensor<float> x_copy({2, 4});
    Tensor<float> out({2, 4});
    fill_pattern(x);
    copy_tensor(x, x_copy);
    softmax(x, out);
    expect_allclose(x, x_copy);
}

TEST(Softmax, AllocatingWrapper) {
    Tensor<float> x({3});
    Tensor<float> want({3});
    fill_iota(x, 0.5f);
    Tensor<float> out = softmax(x);
    EXPECT_EQ(out.shape(), x.shape());
    softmax_ref(x, want);
    expect_allclose(out, want);
}

TEST(Softmax, ShapeMismatchThrows) {
    Tensor<float> x({2, 3});
    Tensor<float> out_bad({3, 2});
    EXPECT_THROW(softmax(x, out_bad), std::invalid_argument);
}

TEST(Softmax, EmptyThrows) {
    Tensor<float> x;
    Tensor<float> out;
    EXPECT_THROW(softmax(x, out), std::invalid_argument);
    EXPECT_THROW(softmax(x), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// QK^T
// ---------------------------------------------------------------------------

TEST(AttentionQK, KnownValues) {
    Tensor<float> q({2, 2});
    Tensor<float> k({2, 2});
    Tensor<float> scores({2, 2});
    q.at({0, 0}) = 1.0f;
    q.at({0, 1}) = 0.0f;
    q.at({1, 0}) = 0.0f;
    q.at({1, 1}) = 1.0f;
    k.at({0, 0}) = 1.0f;
    k.at({0, 1}) = 0.0f;
    k.at({1, 0}) = 0.0f;
    k.at({1, 1}) = 1.0f;

    attention_qk(q, k, scores);

    EXPECT_NEAR(scores.at({0, 0}), 1.0f, kTol);
    EXPECT_NEAR(scores.at({0, 1}), 0.0f, kTol);
    EXPECT_NEAR(scores.at({1, 0}), 0.0f, kTol);
    EXPECT_NEAR(scores.at({1, 1}), 1.0f, kTol);
}

TEST(AttentionQK, MatchesIndependentReference) {
    const std::vector<std::vector<size_t>> q_shapes = {{3, 4}, {2, 3, 4}, {2, 2, 3, 8}};
    for (const auto& q_shape : q_shapes) {
        auto k_shape = q_shape;
        auto sc_shape = q_shape;
        k_shape[k_shape.size() - 2] = q_shape[q_shape.size() - 2] + 1;
        sc_shape.back() = k_shape[k_shape.size() - 2];

        Tensor<float> q(q_shape);
        Tensor<float> k(k_shape);
        Tensor<float> got(sc_shape);
        Tensor<float> want(sc_shape);
        fill_pattern(q);
        fill_iota(k, -0.5f);

        attention_qk(q, k, got);
        attention_qk_ref(q, k, want);
        expect_allclose(got, want);
    }
}

TEST(AttentionQK, ShapeMismatchThrows) {
    Tensor<float> q({2, 4});
    Tensor<float> k({2, 4});
    Tensor<float> scores({2, 2});
    Tensor<float> scores_bad({3, 2});
    Tensor<float> k_bad({2, 8});
    Tensor<float> rank1({4});
    EXPECT_THROW(attention_qk(q, k, scores_bad), std::invalid_argument);
    EXPECT_THROW(attention_qk(q, k_bad, scores), std::invalid_argument);
    EXPECT_THROW(attention_qk(rank1, k, scores), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Causal mask
// ---------------------------------------------------------------------------

TEST(AttentionMask, SquareIsLowerTriangular) {
    Tensor<float> scores({3, 3});
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] = 1.0f;
    }

    attention_causal_mask(scores);

    EXPECT_NEAR(scores.at({0, 0}), 1.0f, kTol);
    EXPECT_TRUE(std::isinf(scores.at({0, 1})) && scores.at({0, 1}) < 0.0f);
    EXPECT_TRUE(std::isinf(scores.at({0, 2})) && scores.at({0, 2}) < 0.0f);
    EXPECT_NEAR(scores.at({1, 0}), 1.0f, kTol);
    EXPECT_NEAR(scores.at({1, 1}), 1.0f, kTol);
    EXPECT_TRUE(std::isinf(scores.at({1, 2})) && scores.at({1, 2}) < 0.0f);
    EXPECT_NEAR(scores.at({2, 0}), 1.0f, kTol);
    EXPECT_NEAR(scores.at({2, 1}), 1.0f, kTol);
    EXPECT_NEAR(scores.at({2, 2}), 1.0f, kTol);
}

TEST(AttentionMask, DecodeQuerySeesAllKeys) {
    Tensor<float> scores({1, 4});
    fill_iota(scores, 1.0f);
    Tensor<float> before({1, 4});
    copy_tensor(scores, before);

    attention_causal_mask(scores);
    expect_allclose(scores, before);
}

TEST(AttentionMask, BottomRightAlignment) {
    Tensor<float> scores({2, 4});
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] = 1.0f;
    }
    Tensor<float> want({2, 4});
    copy_tensor(scores, want);
    attention_causal_mask_ref(want);

    attention_causal_mask(scores);
    expect_allclose(scores, want);

    // Query 0 (absolute pos 2) cannot see key 3.
    EXPECT_TRUE(std::isinf(scores.at({0, 3})) && scores.at({0, 3}) < 0.0f);
    EXPECT_NEAR(scores.at({0, 2}), 1.0f, kTol);
    EXPECT_NEAR(scores.at({1, 3}), 1.0f, kTol);
}

TEST(AttentionMask, BroadcastsOverHeads) {
    Tensor<float> scores({2, 3, 3});
    fill_iota(scores, 0.0f);
    Tensor<float> want({2, 3, 3});
    copy_tensor(scores, want);
    attention_causal_mask_ref(want);

    attention_causal_mask(scores);
    expect_allclose(scores, want);
}

// ---------------------------------------------------------------------------
// Attention × V
// ---------------------------------------------------------------------------

TEST(AttentionAV, OneHotPicksValueRow) {
    Tensor<float> attn({2, 3});
    Tensor<float> v({3, 4});
    Tensor<float> out({2, 4});
    for (size_t i = 0; i < attn.size(); ++i) {
        attn[i] = 0.0f;
    }
    attn.at({0, 2}) = 1.0f;
    attn.at({1, 0}) = 1.0f;
    fill_iota(v, 10.0f);

    attention_av(attn, v, out);

    for (size_t t = 0; t < 4; ++t) {
        EXPECT_NEAR(out.at({0, t}), v.at({2, t}), kTol);
        EXPECT_NEAR(out.at({1, t}), v.at({0, t}), kTol);
    }
}

TEST(AttentionAV, MatchesIndependentReference) {
    Tensor<float> attn({2, 3, 5});
    Tensor<float> v({2, 5, 4});
    Tensor<float> got({2, 3, 4});
    Tensor<float> want({2, 3, 4});
    fill_pattern(attn);
    fill_iota(v, 0.25f);

    attention_av(attn, v, got);
    attention_av_ref(attn, v, want);
    expect_allclose(got, want);
}

TEST(AttentionAV, ShapeMismatchThrows) {
    Tensor<float> attn({2, 3});
    Tensor<float> v({3, 4});
    Tensor<float> out({2, 4});
    Tensor<float> out_bad({2, 3});
    Tensor<float> v_bad({2, 4});
    EXPECT_THROW(attention_av(attn, v, out_bad), std::invalid_argument);
    EXPECT_THROW(attention_av(attn, v_bad, out), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Full scaled dot-product attention
// ---------------------------------------------------------------------------

TEST(Attention, CausalLowerTriangularWhenVIsIdentity) {
    const size_t seq = 4;
    const size_t d = 4;
    Tensor<float> q({seq, d});
    Tensor<float> k({seq, d});
    Tensor<float> v({seq, seq});
    Tensor<float> out({seq, seq});
    fill_pattern(q);
    fill_iota(k, -1.0f);
    for (size_t i = 0; i < seq; ++i) {
        for (size_t j = 0; j < seq; ++j) {
            v.at({i, j}) = (i == j) ? 1.0f : 0.0f;
        }
    }

    attention(q, k, v, out, /*causal=*/true);

    for (size_t i = 0; i < seq; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < seq; ++j) {
            if (j > i) {
                EXPECT_NEAR(out.at({i, j}), 0.0f, kTol) << "future i=" << i << " j=" << j;
            } else {
                EXPECT_GE(out.at({i, j}), -kTol);
            }
            row_sum += out.at({i, j});
        }
        EXPECT_NEAR(row_sum, 1.0f, kTol) << "row " << i;
    }
}

TEST(Attention, NonCausalDiffersFromCausal) {
    Tensor<float> q({3, 4});
    Tensor<float> k({3, 4});
    Tensor<float> v({3, 4});
    Tensor<float> causal({3, 4});
    Tensor<float> full({3, 4});
    fill_pattern(q);
    fill_iota(k, 0.5f);
    fill_iota(v, -2.0f);

    attention(q, k, v, causal, /*causal=*/true);
    attention(q, k, v, full, /*causal=*/false);

    bool differs = false;
    for (size_t i = 0; i < causal.size(); ++i) {
        if (std::abs(causal[i] - full[i]) > kTol) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs);
}

TEST(Attention, MultiHeadIndependent) {
    Tensor<float> q({2, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> v({2, 3, 4});
    Tensor<float> out({2, 3, 4});
    fill_iota(q, 0.25f);
    fill_pattern(k);
    fill_iota(v, -1.0f);

    attention(q, k, v, out, /*causal=*/true);

    for (size_t h = 0; h < 2; ++h) {
        Tensor<float> qh({3, 4});
        Tensor<float> kh({3, 4});
        Tensor<float> vh({3, 4});
        Tensor<float> oh({3, 4});
        copy_head(q, h, qh);
        copy_head(k, h, kh);
        copy_head(v, h, vh);
        attention_ref(qh, kh, vh, oh, true);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 4; ++t) {
                EXPECT_NEAR(out.at({h, i, t}), oh.at({i, t}), kTol) << "h=" << h;
            }
        }
    }
}

TEST(Attention, MatchesIndependentReference) {
    struct Case {
        std::vector<size_t> qkv;
        bool causal;
        float scale;
    };
    const std::vector<Case> cases = {
        {{2, 4}, true, -1.0f},
        {{4, 8}, false, -1.0f},
        {{3, 2, 4}, true, -1.0f},
        {{2, 3, 8}, false, 0.5f},
        {{2, 2, 3, 4}, true, -1.0f},
    };

    for (const auto& c : cases) {
        Tensor<float> q(c.qkv);
        Tensor<float> k(c.qkv);
        Tensor<float> v(c.qkv);
        Tensor<float> got(c.qkv);
        Tensor<float> want(c.qkv);
        fill_pattern(q);
        fill_iota(k, 0.1f);
        fill_iota(v, -0.3f);

        attention(q, k, v, got, c.causal, c.scale);
        attention_ref(q, k, v, want, c.causal, c.scale);
        // matmul vs the nested-loop ref can differ by ~1e-5 in fp32 reduction order
        expect_allclose(got, want, 2e-5f);
    }
}

TEST(Attention, DecodeSingleQuery) {
    Tensor<float> q({1, 4});
    Tensor<float> k({5, 4});
    Tensor<float> v({5, 4});
    Tensor<float> got_causal({1, 4});
    Tensor<float> got_full({1, 4});
    Tensor<float> want({1, 4});
    fill_iota(q, 1.0f);
    fill_pattern(k);
    fill_iota(v, 0.5f);

    attention(q, k, v, got_causal, /*causal=*/true);
    attention(q, k, v, got_full, /*causal=*/false);
    attention_ref(q, k, v, want, true);

    expect_allclose(got_causal, want);
    expect_allclose(got_causal, got_full);
}

TEST(Attention, CustomScale) {
    Tensor<float> q({3, 4});
    Tensor<float> k({3, 4});
    Tensor<float> v({3, 4});
    Tensor<float> got({3, 4});
    Tensor<float> want({3, 4});
    fill_pattern(q);
    fill_iota(k, -0.2f);
    fill_iota(v, 1.5f);

    attention(q, k, v, got, /*causal=*/true, /*scale=*/1.0f);
    attention_ref(q, k, v, want, true, 1.0f);
    expect_allclose(got, want);
}

TEST(Attention, DefaultScaleIsInvSqrtD) {
    Tensor<float> q({3, 8});
    Tensor<float> k({3, 8});
    Tensor<float> v({3, 8});
    Tensor<float> with_default({3, 8});
    Tensor<float> with_explicit({3, 8});
    fill_pattern(q);
    fill_iota(k, 0.3f);
    fill_iota(v, -0.7f);

    const float scale = 1.0f / std::sqrt(8.0f);
    attention(q, k, v, with_default, /*causal=*/true);
    attention(q, k, v, with_explicit, /*causal=*/true, scale);
    expect_allclose(with_default, with_explicit);
}

TEST(Attention, OverwritesOutput) {
    Tensor<float> q({2, 4});
    Tensor<float> k({2, 4});
    Tensor<float> v({2, 4});
    Tensor<float> out({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(q, -1.0f);
    fill_pattern(k);
    fill_iota(v, 2.0f);
    fill_iota(out, 99.0f);

    attention(q, k, v, out, /*causal=*/true);
    attention_ref(q, k, v, want, true);
    expect_allclose(out, want);
}

TEST(Attention, LeavesInputsUnchanged) {
    Tensor<float> q({2, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> v({2, 3, 4});
    Tensor<float> q_copy({2, 3, 4});
    Tensor<float> k_copy({2, 3, 4});
    Tensor<float> v_copy({2, 3, 4});
    Tensor<float> out({2, 3, 4});
    fill_pattern(q);
    fill_iota(k, 0.5f);
    fill_iota(v, -1.0f);
    copy_tensor(q, q_copy);
    copy_tensor(k, k_copy);
    copy_tensor(v, v_copy);

    attention(q, k, v, out, /*causal=*/true);
    expect_allclose(q, q_copy);
    expect_allclose(k, k_copy);
    expect_allclose(v, v_copy);
}

TEST(Attention, AllocatingWrapper) {
    Tensor<float> q({2, 4});
    Tensor<float> k({2, 4});
    Tensor<float> v({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(q, 0.25f);
    fill_pattern(k);
    fill_iota(v, -0.5f);

    Tensor<float> out = attention(q, k, v, /*causal=*/true);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 4}));
    attention_ref(q, k, v, want, true);
    expect_allclose(out, want);
}

TEST(Attention, ContiguousReshapeView) {
    Tensor<float> q_store({24});
    Tensor<float> k_store({24});
    Tensor<float> v_store({24});
    fill_iota(q_store, 1.0f);
    fill_pattern(k_store);
    fill_iota(v_store, -2.0f);
    Tensor<float> q = q_store.view({2, 3, 4});
    Tensor<float> k = k_store.view({2, 3, 4});
    Tensor<float> v = v_store.view({2, 3, 4});
    Tensor<float> out({2, 3, 4});
    Tensor<float> want({2, 3, 4});

    attention(q, k, v, out, /*causal=*/true);
    attention_ref(q, k, v, want, true);
    expect_allclose(out, want);
}

TEST(Attention, ScaleHelper) {
    Tensor<float> scores({2, 2});
    fill_iota(scores, 1.0f);
    attention_scale(scores, 0.5f);
    EXPECT_NEAR(scores[0], 0.5f, kTol);
    EXPECT_NEAR(scores[1], 1.0f, kTol);
    EXPECT_NEAR(scores[2], 1.5f, kTol);
    EXPECT_NEAR(scores[3], 2.0f, kTol);
}

TEST(Attention, ShapeMismatchThrows) {
    Tensor<float> q({2, 4});
    Tensor<float> k({2, 4});
    Tensor<float> v({2, 4});
    Tensor<float> out({2, 4});
    Tensor<float> out_bad({3, 4});
    Tensor<float> k_bad({2, 8});
    Tensor<float> v_bad({3, 4});
    Tensor<float> rank1({8});

    EXPECT_THROW(attention(q, k, v, out_bad), std::invalid_argument);
    EXPECT_THROW(attention(q, k_bad, v, out), std::invalid_argument);
    EXPECT_THROW(attention(q, k, v_bad, out), std::invalid_argument);
    EXPECT_THROW(attention(rank1, k, v, out), std::invalid_argument);
}

TEST(Attention, EmptyThrows) {
    Tensor<float> q;
    Tensor<float> k({2, 4});
    Tensor<float> v({2, 4});
    Tensor<float> out({2, 4});
    EXPECT_THROW(attention(q, k, v, out), std::invalid_argument);
    EXPECT_THROW(attention(q, k, v), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Prefill vs decode (MHA) — no KV cache; decode is seq_q=1 against a prefix
// ---------------------------------------------------------------------------

TEST(Attention, PrefillLastRowEqualsDecode) {
    const size_t seq = 6;
    const size_t d = 4;
    Tensor<float> q({seq, d});
    Tensor<float> k({seq, d});
    Tensor<float> v({seq, d});
    Tensor<float> prefill({seq, d});
    fill_pattern(q);
    fill_iota(k, 0.25f);
    fill_iota(v, -1.0f);

    attention(q, k, v, prefill, /*causal=*/true);

    Tensor<float> q_dec({1, d});
    Tensor<float> got({1, d});
    copy_query_at(q, seq - 1, q_dec);
    attention(q_dec, k, v, got, /*causal=*/true);

    for (size_t t = 0; t < d; ++t) {
        EXPECT_NEAR(got.at({0, t}), prefill.at({seq - 1, t}), 2e-5f);
    }
}

TEST(Attention, PrefillEveryPositionEqualsDecode) {
    const size_t seq = 5;
    const size_t d = 8;
    Tensor<float> q({2, seq, d});
    Tensor<float> k({2, seq, d});
    Tensor<float> v({2, seq, d});
    Tensor<float> prefill({2, seq, d});
    fill_iota(q, 0.1f);
    fill_pattern(k);
    fill_iota(v, -0.4f);

    attention(q, k, v, prefill, /*causal=*/true);

    for (size_t pos = 0; pos < seq; ++pos) {
        Tensor<float> q_dec({2, 1, d});
        Tensor<float> k_pref({2, pos + 1, d});
        Tensor<float> v_pref({2, pos + 1, d});
        Tensor<float> got({2, 1, d});
        copy_query_at(q, pos, q_dec);
        copy_seq_prefix(k, k_pref);
        copy_seq_prefix(v, v_pref);
        attention(q_dec, k_pref, v_pref, got, /*causal=*/true);
        for (size_t h = 0; h < 2; ++h) {
            for (size_t t = 0; t < d; ++t) {
                EXPECT_NEAR(got.at({h, 0, t}), prefill.at({h, pos, t}), 2e-5f)
                    << "h=" << h << " pos=" << pos;
            }
        }
    }
}

TEST(Attention, DecodeCausalEqualsNonCausal) {
    Tensor<float> q({3, 1, 4});
    Tensor<float> k({3, 7, 4});
    Tensor<float> v({3, 7, 4});
    Tensor<float> causal({3, 1, 4});
    Tensor<float> full({3, 1, 4});
    fill_pattern(q);
    fill_iota(k, -0.5f);
    fill_iota(v, 1.25f);

    attention(q, k, v, causal, /*causal=*/true);
    attention(q, k, v, full, /*causal=*/false);
    expect_allclose(causal, full, 2e-5f);
}

TEST(Attention, ChunkPrefillBottomRight) {
    // Queries for the last 2 tokens of a 5-token context.
    Tensor<float> q_full({5, 4});
    Tensor<float> k({5, 4});
    Tensor<float> v({5, 4});
    Tensor<float> prefill({5, 4});
    fill_iota(q_full, 0.2f);
    fill_pattern(k);
    fill_iota(v, -0.8f);
    attention(q_full, k, v, prefill, /*causal=*/true);

    Tensor<float> q_chunk({2, 4});
    Tensor<float> got({2, 4});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t t = 0; t < 4; ++t) {
            q_chunk.at({i, t}) = q_full.at({3 + i, t});
        }
    }
    attention(q_chunk, k, v, got, /*causal=*/true);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t t = 0; t < 4; ++t) {
            EXPECT_NEAR(got.at({i, t}), prefill.at({3 + i, t}), 2e-5f);
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped-query attention
// ---------------------------------------------------------------------------

TEST(AttentionQK, GqaRepeatsKvHeads) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({2, 5, 4});
    Tensor<float> got({8, 3, 5});
    Tensor<float> want({8, 3, 5});
    fill_pattern(q);
    fill_iota(k, -0.5f);

    attention_qk(q, k, got);
    Tensor<float> k_exp = expand_kv_heads(k, 8);
    attention_qk_ref(q, k_exp, want);
    expect_allclose(got, want);
}

TEST(AttentionQK, GqaBatched) {
    Tensor<float> q({2, 8, 3, 4});
    Tensor<float> k({2, 2, 5, 4});
    Tensor<float> got({2, 8, 3, 5});
    Tensor<float> want({2, 8, 3, 5});
    fill_iota(q, 0.25f);
    fill_pattern(k);

    attention_qk(q, k, got);
    Tensor<float> k_exp = expand_kv_heads(k, 8);
    attention_qk_ref(q, k_exp, want);
    expect_allclose(got, want);
}

TEST(AttentionAV, GqaRepeatsKvHeads) {
    Tensor<float> attn({8, 3, 5});
    Tensor<float> v({2, 5, 4});
    Tensor<float> got({8, 3, 4});
    Tensor<float> want({8, 3, 4});
    fill_pattern(attn);
    fill_iota(v, 0.25f);

    attention_av(attn, v, got);
    Tensor<float> v_exp = expand_kv_heads(v, 8);
    attention_av_ref(attn, v_exp, want);
    expect_allclose(got, want);
}

TEST(AttentionGqa, MatchesExpandedMhaPrefill) {
    Tensor<float> q({8, 6, 4});
    Tensor<float> k({2, 6, 4});
    Tensor<float> v({2, 6, 4});
    Tensor<float> got({8, 6, 4});
    Tensor<float> want({8, 6, 4});
    fill_pattern(q);
    fill_iota(k, 0.1f);
    fill_iota(v, -0.3f);

    attention(q, k, v, got, /*causal=*/true);
    attention_gqa_ref(q, k, v, want, true);
    expect_allclose(got, want, 2e-5f);
}

TEST(AttentionGqa, NonCausalDiffersFromCausalPrefill) {
    Tensor<float> q({4, 5, 8});
    Tensor<float> k({2, 5, 8});
    Tensor<float> v({2, 5, 8});
    Tensor<float> causal({4, 5, 8});
    Tensor<float> full({4, 5, 8});
    fill_iota(q, 0.4f);
    fill_pattern(k);
    fill_iota(v, -1.1f);

    attention(q, k, v, causal, /*causal=*/true);
    attention(q, k, v, full, /*causal=*/false);

    bool differs = false;
    for (size_t i = 0; i < causal.size(); ++i) {
        if (std::abs(causal[i] - full[i]) > kTol) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs);
}

TEST(AttentionGqa, Mqa) {
    Tensor<float> q({4, 5, 8});
    Tensor<float> k({1, 5, 8});
    Tensor<float> v({1, 5, 8});
    Tensor<float> got({4, 5, 8});
    Tensor<float> want({4, 5, 8});
    fill_pattern(q);
    fill_iota(k, -0.2f);
    fill_iota(v, 0.7f);

    attention(q, k, v, got, /*causal=*/true);
    attention_gqa_ref(q, k, v, want, true);
    expect_allclose(got, want, 2e-5f);
}

TEST(AttentionGqa, BatchedPrefill) {
    Tensor<float> q({3, 8, 4, 8});
    Tensor<float> k({3, 2, 4, 8});
    Tensor<float> v({3, 2, 4, 8});
    Tensor<float> got({3, 8, 4, 8});
    Tensor<float> want({3, 8, 4, 8});
    fill_iota(q, 0.15f);
    fill_pattern(k);
    fill_iota(v, -0.55f);

    attention(q, k, v, got, /*causal=*/true);
    attention_gqa_ref(q, k, v, want, true);
    expect_allclose(got, want, 2e-5f);
}

TEST(AttentionGqa, GroupSharesKvHead) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> scores({8, 3, 3});
    fill_iota(q, 1.0f);
    fill_pattern(k);

    // Heads 0 and 1 are in the same group (group size 4).
    for (size_t i = 0; i < 3; ++i) {
        for (size_t t = 0; t < 4; ++t) {
            q.at({1, i, t}) = q.at({0, i, t});
        }
    }
    attention_qk(q, k, scores);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(scores.at({0, i, j}), scores.at({1, i, j}), kTol);
        }
    }

    // Head 4 is the first head of the second KV group — different K.
    bool differs = false;
    for (size_t i = 0; i < 3 && !differs; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (std::abs(scores.at({0, i, j}) - scores.at({4, i, j})) > kTol) {
                differs = true;
                break;
            }
        }
    }
    EXPECT_TRUE(differs);
}

TEST(AttentionGqa, DecodeSingleQuery) {
    Tensor<float> q({8, 1, 4});
    Tensor<float> k({2, 6, 4});
    Tensor<float> v({2, 6, 4});
    Tensor<float> got_causal({8, 1, 4});
    Tensor<float> got_full({8, 1, 4});
    Tensor<float> want({8, 1, 4});
    fill_iota(q, 1.0f);
    fill_pattern(k);
    fill_iota(v, 0.5f);

    attention(q, k, v, got_causal, /*causal=*/true);
    attention(q, k, v, got_full, /*causal=*/false);
    attention_gqa_ref(q, k, v, want, true);

    expect_allclose(got_causal, want, 2e-5f);
    expect_allclose(got_causal, got_full, 2e-5f);
}

TEST(AttentionGqa, PrefillLastRowEqualsDecode) {
    const size_t n_q = 8;
    const size_t n_kv = 2;
    const size_t seq = 6;
    const size_t d = 4;
    Tensor<float> q({n_q, seq, d});
    Tensor<float> k({n_kv, seq, d});
    Tensor<float> v({n_kv, seq, d});
    Tensor<float> prefill({n_q, seq, d});
    fill_pattern(q);
    fill_iota(k, 0.25f);
    fill_iota(v, -1.0f);

    attention(q, k, v, prefill, /*causal=*/true);

    Tensor<float> q_dec({n_q, 1, d});
    Tensor<float> got({n_q, 1, d});
    copy_query_at(q, seq - 1, q_dec);
    attention(q_dec, k, v, got, /*causal=*/true);

    for (size_t h = 0; h < n_q; ++h) {
        for (size_t t = 0; t < d; ++t) {
            EXPECT_NEAR(got.at({h, 0, t}), prefill.at({h, seq - 1, t}), 2e-5f) << "h=" << h;
        }
    }
}

TEST(AttentionGqa, PrefillEveryPositionEqualsDecode) {
    const size_t n_q = 8;
    const size_t n_kv = 2;
    const size_t seq = 5;
    const size_t d = 8;
    Tensor<float> q({n_q, seq, d});
    Tensor<float> k({n_kv, seq, d});
    Tensor<float> v({n_kv, seq, d});
    Tensor<float> prefill({n_q, seq, d});
    fill_iota(q, 0.1f);
    fill_pattern(k);
    fill_iota(v, -0.4f);

    attention(q, k, v, prefill, /*causal=*/true);

    for (size_t pos = 0; pos < seq; ++pos) {
        Tensor<float> q_dec({n_q, 1, d});
        Tensor<float> k_pref({n_kv, pos + 1, d});
        Tensor<float> v_pref({n_kv, pos + 1, d});
        Tensor<float> got({n_q, 1, d});
        copy_query_at(q, pos, q_dec);
        copy_seq_prefix(k, k_pref);
        copy_seq_prefix(v, v_pref);

        attention(q_dec, k_pref, v_pref, got, /*causal=*/true);
        Tensor<float> want({n_q, 1, d});
        attention_gqa_ref(q_dec, k_pref, v_pref, want, true);
        expect_allclose(got, want, 2e-5f);

        for (size_t h = 0; h < n_q; ++h) {
            for (size_t t = 0; t < d; ++t) {
                EXPECT_NEAR(got.at({h, 0, t}), prefill.at({h, pos, t}), 2e-5f)
                    << "h=" << h << " pos=" << pos;
            }
        }
    }
}

TEST(AttentionGqa, PrefillThenMultiStepDecode) {
    const size_t n_q = 4;
    const size_t n_kv = 1;
    const size_t prompt = 4;
    const size_t gen = 3;
    const size_t T = prompt + gen;
    const size_t d = 4;

    Tensor<float> q_full({n_q, T, d});
    Tensor<float> k_full({n_kv, T, d});
    Tensor<float> v_full({n_kv, T, d});
    fill_pattern(q_full);
    fill_iota(k_full, 0.3f);
    fill_iota(v_full, -0.6f);

    Tensor<float> q_prompt(with_seq({n_q, T, d}, prompt));
    Tensor<float> k_prompt(with_seq({n_kv, T, d}, prompt));
    Tensor<float> v_prompt(with_seq({n_kv, T, d}, prompt));
    Tensor<float> prefill(with_seq({n_q, T, d}, prompt));
    copy_seq_prefix(q_full, q_prompt);
    copy_seq_prefix(k_full, k_prompt);
    copy_seq_prefix(v_full, v_prompt);
    attention(q_prompt, k_prompt, v_prompt, prefill, /*causal=*/true);

    Tensor<float> want_prefill(with_seq({n_q, T, d}, prompt));
    attention_gqa_ref(q_prompt, k_prompt, v_prompt, want_prefill, true);
    expect_allclose(prefill, want_prefill, 2e-5f);

    Tensor<float> full({n_q, T, d});
    attention(q_full, k_full, v_full, full, /*causal=*/true);

    for (size_t i = 0; i < prompt; ++i) {
        for (size_t h = 0; h < n_q; ++h) {
            for (size_t t = 0; t < d; ++t) {
                EXPECT_NEAR(prefill.at({h, i, t}), full.at({h, i, t}), 2e-5f)
                    << "prompt pos=" << i;
            }
        }
    }

    for (size_t step = 0; step < gen; ++step) {
        const size_t pos = prompt + step;
        Tensor<float> q_dec({n_q, 1, d});
        Tensor<float> k_cache({n_kv, pos + 1, d});
        Tensor<float> v_cache({n_kv, pos + 1, d});
        Tensor<float> got({n_q, 1, d});
        copy_query_at(q_full, pos, q_dec);
        copy_seq_prefix(k_full, k_cache);
        copy_seq_prefix(v_full, v_cache);
        attention(q_dec, k_cache, v_cache, got, /*causal=*/true);

        for (size_t h = 0; h < n_q; ++h) {
            for (size_t t = 0; t < d; ++t) {
                EXPECT_NEAR(got.at({h, 0, t}), full.at({h, pos, t}), 2e-5f)
                    << "decode step=" << step;
            }
        }
    }
}

TEST(AttentionGqa, BatchedDecode) {
    Tensor<float> q({2, 8, 1, 4});
    Tensor<float> k({2, 2, 7, 4});
    Tensor<float> v({2, 2, 7, 4});
    Tensor<float> got({2, 8, 1, 4});
    Tensor<float> want({2, 8, 1, 4});
    fill_iota(q, 0.9f);
    fill_pattern(k);
    fill_iota(v, -0.15f);

    attention(q, k, v, got, /*causal=*/true);
    attention_gqa_ref(q, k, v, want, true);
    expect_allclose(got, want, 2e-5f);
}

TEST(AttentionGqa, ChunkPrefillBottomRight) {
    Tensor<float> q_full({8, 6, 4});
    Tensor<float> k({2, 6, 4});
    Tensor<float> v({2, 6, 4});
    Tensor<float> prefill({8, 6, 4});
    fill_iota(q_full, 0.2f);
    fill_pattern(k);
    fill_iota(v, -0.8f);
    attention(q_full, k, v, prefill, /*causal=*/true);

    Tensor<float> q_chunk({8, 2, 4});
    Tensor<float> got({8, 2, 4});
    for (size_t h = 0; h < 8; ++h) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t t = 0; t < 4; ++t) {
                q_chunk.at({h, i, t}) = q_full.at({h, 4 + i, t});
            }
        }
    }
    attention(q_chunk, k, v, got, /*causal=*/true);
    for (size_t h = 0; h < 8; ++h) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t t = 0; t < 4; ++t) {
                EXPECT_NEAR(got.at({h, i, t}), prefill.at({h, 4 + i, t}), 2e-5f);
            }
        }
    }
}

TEST(AttentionGqa, DifferentValueDim) {
    Tensor<float> q({8, 4, 8});
    Tensor<float> k({2, 4, 8});
    Tensor<float> v({2, 4, 16});
    Tensor<float> got({8, 4, 16});
    Tensor<float> want({8, 4, 16});
    fill_pattern(q);
    fill_iota(k, 0.05f);
    fill_iota(v, -0.2f);

    attention(q, k, v, got, /*causal=*/true);
    attention_gqa_ref(q, k, v, want, true);
    expect_allclose(got, want, 2e-5f);
}

TEST(AttentionGqa, CustomScale) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> v({2, 3, 4});
    Tensor<float> got({8, 3, 4});
    Tensor<float> want({8, 3, 4});
    fill_pattern(q);
    fill_iota(k, -0.2f);
    fill_iota(v, 1.5f);

    attention(q, k, v, got, /*causal=*/true, /*scale=*/1.0f);
    attention_gqa_ref(q, k, v, want, true, 1.0f);
    expect_allclose(got, want, 2e-5f);
}

TEST(AttentionGqa, AllocatingWrapper) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> v({2, 3, 4});
    Tensor<float> want({8, 3, 4});
    fill_iota(q, 0.25f);
    fill_pattern(k);
    fill_iota(v, -0.5f);

    Tensor<float> out = attention(q, k, v, /*causal=*/true);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{8, 3, 4}));
    attention_gqa_ref(q, k, v, want, true);
    expect_allclose(out, want, 2e-5f);
}

TEST(AttentionGqa, LeavesInputsUnchanged) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> v({2, 3, 4});
    Tensor<float> q_copy({8, 3, 4});
    Tensor<float> k_copy({2, 3, 4});
    Tensor<float> v_copy({2, 3, 4});
    Tensor<float> out({8, 3, 4});
    fill_pattern(q);
    fill_iota(k, 0.5f);
    fill_iota(v, -1.0f);
    copy_tensor(q, q_copy);
    copy_tensor(k, k_copy);
    copy_tensor(v, v_copy);

    attention(q, k, v, out, /*causal=*/true);
    expect_allclose(q, q_copy);
    expect_allclose(k, k_copy);
    expect_allclose(v, v_copy);
}

TEST(AttentionGqa, NqEqualsNkvIsMha) {
    Tensor<float> q({4, 3, 8});
    Tensor<float> k({4, 3, 8});
    Tensor<float> v({4, 3, 8});
    Tensor<float> gqa({4, 3, 8});
    Tensor<float> mha({4, 3, 8});
    fill_pattern(q);
    fill_iota(k, 0.1f);
    fill_iota(v, -0.3f);

    attention(q, k, v, gqa, /*causal=*/true);
    attention_ref(q, k, v, mha, true);
    expect_allclose(gqa, mha, 2e-5f);
}

TEST(AttentionGqa, NqNotDivisibleByNkvThrows) {
    Tensor<float> q({6, 3, 4});
    Tensor<float> k({4, 3, 4});
    Tensor<float> v({4, 3, 4});
    Tensor<float> out({6, 3, 4});
    fill_pattern(q);
    fill_iota(k, 0.1f);
    fill_iota(v, 0.2f);
    EXPECT_THROW(attention(q, k, v, out), std::invalid_argument);
}

TEST(AttentionGqa, KvHeadMismatchThrows) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({2, 3, 4});
    Tensor<float> v({4, 3, 4});
    Tensor<float> out({8, 3, 4});
    EXPECT_THROW(attention(q, k, v, out), std::invalid_argument);
}

TEST(AttentionGqa, BatchMismatchThrows) {
    Tensor<float> q({2, 8, 3, 4});
    Tensor<float> k({3, 2, 3, 4});
    Tensor<float> v({3, 2, 3, 4});
    Tensor<float> out({2, 8, 3, 4});
    EXPECT_THROW(attention(q, k, v, out), std::invalid_argument);
}

TEST(AttentionGqa, RankMismatchThrows) {
    Tensor<float> q({8, 3, 4});
    Tensor<float> k({3, 4});
    Tensor<float> v({3, 4});
    Tensor<float> out({8, 3, 4});
    EXPECT_THROW(attention(q, k, v, out), std::invalid_argument);
}
