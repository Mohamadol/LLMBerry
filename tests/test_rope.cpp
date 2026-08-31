#include "llmberry/kernels.h"
#include "llmberry/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using llmberry::rope;
using llmberry::rope_freqs;
using llmberry::rope_sincos;
using llmberry::Tensor;

constexpr float kTheta = 10000.0f;
constexpr float kTol = 1e-5f;

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
        EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
    }
}

float l2_norm(const Tensor<float>& t, size_t offset, size_t dim) {
    float acc = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float v = t[offset + i];
        acc += v * v;
    }
    return std::sqrt(acc);
}

/// Independent inv_freq: 1 / theta^(2i / dim).
void rope_freqs_ref(size_t dim, Tensor<float>& inv_freq, float theta = kTheta) {
    ASSERT_EQ(inv_freq.size(), dim / 2);
    for (size_t i = 0; i < dim / 2; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(dim);
        inv_freq[i] = 1.0f / std::pow(theta, exponent);
    }
}

/// HF Llama tables: concat(freqs, freqs) then cos/sin.
void rope_sincos_ref(const Tensor<float>& inv_freq,
                     const std::vector<size_t>& positions,
                     Tensor<float>& cos,
                     Tensor<float>& sin) {
    const size_t half = inv_freq.size();
    const size_t dim = half * 2;
    ASSERT_EQ(cos.shape(), (std::vector<size_t>{positions.size(), dim}));
    for (size_t t = 0; t < positions.size(); ++t) {
        for (size_t i = 0; i < half; ++i) {
            const float angle = static_cast<float>(positions[t]) * inv_freq[i];
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            cos.at({t, i}) = c;
            cos.at({t, i + half}) = c;
            sin.at({t, i}) = s;
            sin.at({t, i + half}) = s;
        }
    }
}

void rope_sincos_ref(size_t seq_len,
                     size_t dim,
                     Tensor<float>& cos,
                     Tensor<float>& sin,
                     size_t position_offset = 0,
                     float theta = kTheta) {
    Tensor<float> inv_freq({dim / 2});
    rope_freqs_ref(dim, inv_freq, theta);
    std::vector<size_t> positions(seq_len);
    for (size_t t = 0; t < seq_len; ++t) {
        positions[t] = position_offset + t;
    }
    rope_sincos_ref(inv_freq, positions, cos, sin);
}

/// Llama / NeoX apply: y = x * cos + rotate_half(x) * sin.
void apply_rope_ref(const Tensor<float>& x,
                    const Tensor<float>& cos,
                    const Tensor<float>& sin,
                    Tensor<float>& out) {
    const size_t seq = x.shape()[0];
    const size_t dim = x.shape().back();
    const size_t half = dim / 2;
    const size_t n_inner = x.size() / seq / dim;
    for (size_t t = 0; t < seq; ++t) {
        for (size_t h = 0; h < n_inner; ++h) {
            const size_t base = (t * n_inner + h) * dim;
            for (size_t i = 0; i < half; ++i) {
                const float x1 = x[base + i];
                const float x2 = x[base + i + half];
                const float c = cos.at({t, i});
                const float s = sin.at({t, i});
                out[base + i] = x1 * c - x2 * s;
                out[base + i + half] = x2 * c + x1 * s;
            }
        }
    }
}

void rope_ref(const Tensor<float>& x,
              Tensor<float>& out,
              size_t position_offset = 0,
              float theta = kTheta) {
    const size_t seq = x.shape()[0];
    const size_t dim = x.shape().back();
    Tensor<float> cos({seq, dim});
    Tensor<float> sin({seq, dim});
    rope_sincos_ref(seq, dim, cos, sin, position_offset, theta);
    apply_rope_ref(x, cos, sin, out);
}

}  // namespace

TEST(RoPEFreqs, KnownValuesDim4) {
    Tensor<float> inv_freq({2});
    rope_freqs(4, inv_freq, kTheta);

    EXPECT_NEAR(inv_freq[0], 1.0f, kTol);
    EXPECT_NEAR(inv_freq[1], 0.01f, kTol);

    Tensor<float> via_wrapper = rope_freqs(4);
    expect_allclose(via_wrapper, inv_freq);

    Tensor<float> llama3({2});
    rope_freqs(4, llama3, 500000.0f);
    EXPECT_NEAR(llama3[0], 1.0f, kTol);
    EXPECT_GT(std::abs(llama3[1] - inv_freq[1]), 1e-3f);
}

TEST(RoPEFreqs, MatchesIndependentReference) {
    for (size_t dim : {2u, 4u, 8u, 16u, 128u}) {
        Tensor<float> got({dim / 2});
        Tensor<float> want({dim / 2});
        rope_freqs(dim, got, kTheta);
        rope_freqs_ref(dim, want, kTheta);
        expect_allclose(got, want);
    }
}

TEST(RoPEFreqs, OddDimThrows) {
    Tensor<float> inv_freq({2});
    EXPECT_THROW(rope_freqs(3, inv_freq, kTheta), std::invalid_argument);
    EXPECT_THROW(rope_freqs(3), std::invalid_argument);
    EXPECT_THROW(rope_freqs(0), std::invalid_argument);
}

TEST(RoPESinCos, PositionZeroIsIdentityTables) {
    const size_t dim = 8;
    Tensor<float> cos({1, dim});
    Tensor<float> sin({1, dim});
    fill_iota(cos, 99.0f);
    fill_iota(sin, -99.0f);

    rope_sincos(1, dim, cos, sin, /*position_offset=*/0, kTheta);

    for (size_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(cos[i], 1.0f, kTol) << "cos index " << i;
        EXPECT_NEAR(sin[i], 0.0f, kTol) << "sin index " << i;
    }
}

TEST(RoPESinCos, ConcatLayoutAndUnitCircle) {
    const size_t seq = 4;
    const size_t dim = 8;
    const size_t half = dim / 2;
    Tensor<float> cos({seq, dim});
    Tensor<float> sin({seq, dim});
    Tensor<float> want_cos({seq, dim});
    Tensor<float> want_sin({seq, dim});

    rope_sincos(seq, dim, cos, sin, /*position_offset=*/0, kTheta);
    rope_sincos_ref(seq, dim, want_cos, want_sin, 0, kTheta);
    expect_allclose(cos, want_cos);
    expect_allclose(sin, want_sin);

    for (size_t t = 0; t < seq; ++t) {
        for (size_t i = 0; i < half; ++i) {
            EXPECT_NEAR(cos.at({t, i}), cos.at({t, i + half}), kTol);
            EXPECT_NEAR(sin.at({t, i}), sin.at({t, i + half}), kTol);
            const float c = cos.at({t, i});
            const float s = sin.at({t, i});
            EXPECT_NEAR(c * c + s * s, 1.0f, kTol) << "t=" << t << " i=" << i;
        }
    }
}

TEST(RoPE, PositionZeroIsIdentity) {
    Tensor<float> x({1, 4});
    Tensor<float> out({1, 4});
    fill_iota(x, -2.0f);

    rope(x, out, /*position_offset=*/0, kTheta);
    expect_allclose(out, x);
}

TEST(RoPE, KnownRotationDim2) {
    // dim=2, theta=10000 → inv_freq = [1]. Position 1 → angle = 1 radian.
    Tensor<float> x({1, 2});
    Tensor<float> out({1, 2});
    x[0] = 1.0f;
    x[1] = 0.0f;

    rope(x, out, /*position_offset=*/1, kTheta);

    EXPECT_NEAR(out[0], std::cos(1.0f), kTol);
    EXPECT_NEAR(out[1], std::sin(1.0f), kTol);

    x[0] = 0.0f;
    x[1] = 1.0f;
    rope(x, out, 1, kTheta);
    EXPECT_NEAR(out[0], -std::sin(1.0f), kTol);
    EXPECT_NEAR(out[1], std::cos(1.0f), kTol);
}

TEST(RoPE, PreservesNorm) {
    Tensor<float> x({3, 2, 8});
    Tensor<float> out({3, 2, 8});
    fill_pattern(x);

    rope(x, out, /*position_offset=*/4, kTheta);

    const size_t dim = 8;
    const size_t n_rows = x.size() / dim;
    for (size_t r = 0; r < n_rows; ++r) {
        EXPECT_NEAR(l2_norm(out, r * dim, dim), l2_norm(x, r * dim, dim), kTol) << "row " << r;
    }
}

TEST(RoPE, HeadsSharePosition) {
    Tensor<float> x({2, 3, 4});
    Tensor<float> out({2, 3, 4});
    fill_iota(x, 0.5f);
    // Same vector on every head at token 1.
    for (size_t h = 0; h < 3; ++h) {
        x.at({1, h, 0}) = 1.0f;
        x.at({1, h, 1}) = 2.0f;
        x.at({1, h, 2}) = 3.0f;
        x.at({1, h, 3}) = 4.0f;
    }

    rope(x, out, /*position_offset=*/0, kTheta);

    for (size_t d = 0; d < 4; ++d) {
        EXPECT_NEAR(out.at({1, 0, d}), out.at({1, 1, d}), kTol);
        EXPECT_NEAR(out.at({1, 0, d}), out.at({1, 2, d}), kTol);
    }
}

TEST(RoPE, MatchesIndependentReference) {
    const std::vector<std::vector<size_t>> shapes = {
        {1, 4},
        {4, 4},
        {3, 2, 8},
        {5, 4, 16},
        {2, 1, 1, 8},
    };

    for (const auto& shape : shapes) {
        Tensor<float> x(shape);
        Tensor<float> got(shape);
        Tensor<float> want(shape);
        fill_pattern(x);

        rope(x, got, /*position_offset=*/0, kTheta);
        rope_ref(x, want, 0, kTheta);
        expect_allclose(got, want);
    }
}

TEST(RoPE, PositionOffset) {
    Tensor<float> x({2, 4});
    Tensor<float> got({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(x, -1.0f);

    rope(x, got, /*position_offset=*/5, kTheta);
    rope_ref(x, want, 5, kTheta);
    expect_allclose(got, want);

    // Tokens at offset 5,6 match a longer sequence's tokens 5,6.
    Tensor<float> long_x({7, 4});
    Tensor<float> long_out({7, 4});
    for (size_t i = 0; i < long_x.size(); ++i) {
        long_x[i] = 0.0f;
    }
    for (size_t d = 0; d < 4; ++d) {
        long_x.at({5, d}) = x.at({0, d});
        long_x.at({6, d}) = x.at({1, d});
    }
    rope(long_x, long_out, /*position_offset=*/0, kTheta);
    for (size_t d = 0; d < 4; ++d) {
        EXPECT_NEAR(got.at({0, d}), long_out.at({5, d}), kTol);
        EXPECT_NEAR(got.at({1, d}), long_out.at({6, d}), kTol);
    }
}

TEST(RoPE, ArbitraryPositions) {
    const size_t dim = 8;
    const std::vector<size_t> positions = {0, 3, 10};
    Tensor<float> inv_freq = rope_freqs(dim, kTheta);
    Tensor<float> inv_ref({dim / 2});
    rope_freqs_ref(dim, inv_ref, kTheta);

    Tensor<float> cos({positions.size(), dim});
    Tensor<float> sin({positions.size(), dim});
    Tensor<float> want_cos({positions.size(), dim});
    Tensor<float> want_sin({positions.size(), dim});

    rope_sincos(inv_freq, positions, cos, sin);
    rope_sincos_ref(inv_ref, positions, want_cos, want_sin);
    expect_allclose(cos, want_cos);
    expect_allclose(sin, want_sin);

    Tensor<float> x({3, 2, dim});
    Tensor<float> got({3, 2, dim});
    Tensor<float> want({3, 2, dim});
    fill_iota(x, 0.25f);

    rope(x, cos, sin, got);
    apply_rope_ref(x, want_cos, want_sin, want);
    expect_allclose(got, want);
}

TEST(RoPE, OverwritesOutput) {
    Tensor<float> x({2, 4});
    Tensor<float> out({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(x, -2.0f);
    fill_iota(out, 100.0f);

    rope(x, out, /*position_offset=*/2, kTheta);
    rope_ref(x, want, 2, kTheta);
    expect_allclose(out, want);
}

TEST(RoPE, LeavesInputsUnchanged) {
    Tensor<float> x({2, 3, 4});
    Tensor<float> x_copy({2, 3, 4});
    Tensor<float> out({2, 3, 4});
    fill_pattern(x);
    copy_tensor(x, x_copy);

    rope(x, out, /*position_offset=*/1, kTheta);
    expect_allclose(x, x_copy);
}

TEST(RoPE, AllocatingWrapper) {
    Tensor<float> x({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(x, -1.0f);

    Tensor<float> out = rope(x, /*position_offset=*/3, kTheta);
    EXPECT_EQ(out.shape(), x.shape());
    rope_ref(x, want, 3, kTheta);
    expect_allclose(out, want);
}

TEST(RoPE, DefaultThetaIs10000) {
    Tensor<float> x({2, 8});
    fill_pattern(x);

    Tensor<float> with_default = rope(x);
    Tensor<float> with_explicit({2, 8});
    Tensor<float> want({2, 8});
    rope(x, with_explicit, /*position_offset=*/0, 10000.0f);
    rope_ref(x, want, 0, 10000.0f);

    expect_allclose(with_default, want);
    expect_allclose(with_explicit, want);
}

TEST(RoPE, ContiguousReshapeView) {
    Tensor<float> storage({24});
    fill_iota(storage, 1.0f);
    Tensor<float> x = storage.view({3, 2, 4});
    Tensor<float> out({3, 2, 4});
    Tensor<float> want({3, 2, 4});

    rope(x, out, /*position_offset=*/0, kTheta);
    rope_ref(x, want, 0, kTheta);
    expect_allclose(out, want);
}

TEST(RoPE, ShapeMismatchThrows) {
    Tensor<float> x({2, 4});
    Tensor<float> out({2, 4});
    Tensor<float> cos({2, 4});
    Tensor<float> sin({2, 4});

    Tensor<float> out_bad({4, 2});
    Tensor<float> cos_bad({3, 4});
    Tensor<float> sin_bad({2, 8});
    Tensor<float> rank1({8});

    EXPECT_THROW(rope(x, cos, sin, out_bad), std::invalid_argument);
    EXPECT_THROW(rope(x, cos_bad, sin, out), std::invalid_argument);
    EXPECT_THROW(rope(x, cos, sin_bad, out), std::invalid_argument);
    EXPECT_THROW(rope(rank1, out), std::invalid_argument);
    EXPECT_THROW(rope(x, out_bad), std::invalid_argument);

    Tensor<float> inv_bad({3});
    EXPECT_THROW(rope_freqs(4, inv_bad, kTheta), std::invalid_argument);
    EXPECT_THROW(rope_sincos(3, 4, cos, sin), std::invalid_argument);
}

TEST(RoPE, EmptyThrows) {
    Tensor<float> x;
    Tensor<float> out;
    EXPECT_THROW(rope(x, out), std::invalid_argument);
    EXPECT_THROW(rope(x), std::invalid_argument);
}

TEST(RoPE, OddHeadDimThrows) {
    Tensor<float> x({2, 3});
    Tensor<float> out({2, 3});
    EXPECT_THROW(rope(x, out), std::invalid_argument);
}
