#include "llmberry/kernels.h"
#include "llmberry/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using llmberry::rmsnorm;
using llmberry::Tensor;

constexpr float kEps = 1e-6f;

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

void expect_allclose(const Tensor<float>& got, const Tensor<float>& want, float tol = 1e-5f) {
    ASSERT_EQ(got.shape(), want.shape());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
    }
}

/// Independent RMSNorm: last-dim mean of squares, then scale by weight.
void rmsnorm_ref(const Tensor<float>& x,
                 const Tensor<float>& weight,
                 Tensor<float>& out,
                 float eps = kEps) {
    const size_t d = weight.size();
    const size_t rows = x.size() / d;
    for (size_t r = 0; r < rows; ++r) {
        float mean_sq = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            const float v = x[r * d + i];
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(d);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (size_t i = 0; i < d; ++i) {
            out[r * d + i] = x[r * d + i] * inv_rms * weight[i];
        }
    }
}

}  // namespace

TEST(RMSNorm, OnesWeightConstantVector) {
    Tensor<float> x({4});
    Tensor<float> w({4});
    Tensor<float> out({4});
    for (size_t i = 0; i < 4; ++i) {
        x[i] = 2.0f;
        w[i] = 1.0f;
    }

    rmsnorm(x, w, out, kEps);

    // mean(x^2) = 4, inv_rms ≈ 1/2, so y ≈ [1, 1, 1, 1]
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f);
    }
}

TEST(RMSNorm, ZeroInputIsZero) {
    Tensor<float> x = Tensor<float>::zeros({3});
    Tensor<float> w = Tensor<float>::ones({3});
    Tensor<float> out({3});
    fill_iota(out, 99.0f);

    rmsnorm(x, w, out, kEps);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}

TEST(RMSNorm, WeightScalesNormalizedOutput) {
    Tensor<float> x({4});
    Tensor<float> w_ones = Tensor<float>::ones({4});
    Tensor<float> w_scale({4});
    Tensor<float> y_ones({4});
    Tensor<float> y_scale({4});
    fill_iota(x, 1.0f);
    w_scale[0] = 0.5f;
    w_scale[1] = 1.0f;
    w_scale[2] = 2.0f;
    w_scale[3] = -1.0f;

    rmsnorm(x, w_ones, y_ones, kEps);
    rmsnorm(x, w_scale, y_scale, kEps);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(y_scale[i], y_ones[i] * w_scale[i], 1e-5f);
    }
}

TEST(RMSNorm, LastDimNormalizedIndependently) {
    Tensor<float> x({2, 3});
    Tensor<float> w = Tensor<float>::ones({3});
    Tensor<float> out({2, 3});
    Tensor<float> want({2, 3});

    // Row 0 is small, row 1 is large — each row has its own RMS.
    x.at({0, 0}) = 1.0f;
    x.at({0, 1}) = 2.0f;
    x.at({0, 2}) = 3.0f;
    x.at({1, 0}) = 10.0f;
    x.at({1, 1}) = 20.0f;
    x.at({1, 2}) = 30.0f;

    rmsnorm(x, w, out, kEps);
    rmsnorm_ref(x, w, want, kEps);
    expect_allclose(out, want);

    // Same direction, different magnitude → same normalized row (weight = 1).
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(out.at({0, i}), out.at({1, i}), 1e-5f);
    }
}

TEST(RMSNorm, MatchesIndependentReference) {
    const std::vector<std::vector<size_t>> shapes = {
        {8},
        {1, 4},
        {3, 5},
        {2, 4, 6},
        {1, 1, 16},
    };

    for (const auto& shape : shapes) {
        Tensor<float> x(shape);
        Tensor<float> w({shape.back()});
        Tensor<float> got(shape);
        Tensor<float> want(shape);
        fill_pattern(x);
        fill_iota(w, 0.25f);

        rmsnorm(x, w, got, kEps);
        rmsnorm_ref(x, w, want, kEps);
        expect_allclose(got, want);
    }
}

TEST(RMSNorm, OverwritesOutput) {
    Tensor<float> x({5});
    Tensor<float> w = Tensor<float>::ones({5});
    Tensor<float> out({5});
    Tensor<float> want({5});
    fill_iota(x, -2.0f);
    fill_iota(out, 100.0f);

    rmsnorm(x, w, out, kEps);
    rmsnorm_ref(x, w, want, kEps);
    expect_allclose(out, want);
}

TEST(RMSNorm, LeavesInputsUnchanged) {
    Tensor<float> x({2, 4});
    Tensor<float> w({4});
    Tensor<float> x_copy({2, 4});
    Tensor<float> w_copy({4});
    Tensor<float> out({2, 4});
    fill_pattern(x);
    fill_iota(w, 0.5f);
    copy_tensor(x, x_copy);
    copy_tensor(w, w_copy);

    rmsnorm(x, w, out, kEps);

    expect_allclose(x, x_copy);
    expect_allclose(w, w_copy);
}

TEST(RMSNorm, AllocatingWrapper) {
    Tensor<float> x({2, 3});
    Tensor<float> w({3});
    Tensor<float> want({2, 3});
    fill_iota(x, -1.0f);
    fill_iota(w, 0.5f);

    Tensor<float> out = rmsnorm(x, w, kEps);
    EXPECT_EQ(out.shape(), x.shape());
    rmsnorm_ref(x, w, want, kEps);
    expect_allclose(out, want);
}

TEST(RMSNorm, DefaultEpsMatchesLlama) {
    Tensor<float> x({4});
    Tensor<float> w = Tensor<float>::ones({4});
    Tensor<float> with_default({4});
    Tensor<float> with_explicit({4});
    fill_pattern(x);

    rmsnorm(x, w, with_explicit, 1e-6f);
    Tensor<float> via_wrapper = rmsnorm(x, w);

    rmsnorm_ref(x, w, with_default, 1e-6f);
    expect_allclose(via_wrapper, with_default);
    expect_allclose(with_explicit, with_default);
}

TEST(RMSNorm, SmallValuesStayFinite) {
    Tensor<float> x({4});
    Tensor<float> w = Tensor<float>::ones({4});
    Tensor<float> out({4});
    for (size_t i = 0; i < 4; ++i) {
        x[i] = 1e-20f;
    }

    rmsnorm(x, w, out, kEps);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "index " << i;
        EXPECT_NEAR(out[i], 0.0f, 1e-5f);
    }
}

TEST(RMSNorm, ContiguousReshapeView) {
    Tensor<float> storage({12});
    fill_iota(storage, 1.0f);
    Tensor<float> x = storage.view({3, 4});
    Tensor<float> w({4});
    Tensor<float> out({3, 4});
    Tensor<float> want({3, 4});
    fill_iota(w, 0.5f);

    rmsnorm(x, w, out, kEps);
    rmsnorm_ref(x, w, want, kEps);
    expect_allclose(out, want);
}

TEST(RMSNorm, ShapeMismatchThrows) {
    Tensor<float> x({2, 4});
    Tensor<float> w({4});
    Tensor<float> out({2, 4});

    Tensor<float> w_bad({3});
    Tensor<float> w_2d({2, 4});
    Tensor<float> out_bad({4, 2});
    Tensor<float> out_flat({8});

    EXPECT_THROW(rmsnorm(x, w_bad, out), std::invalid_argument);
    EXPECT_THROW(rmsnorm(x, w_2d, out), std::invalid_argument);
    EXPECT_THROW(rmsnorm(x, w, out_bad), std::invalid_argument);
    EXPECT_THROW(rmsnorm(x, w, out_flat), std::invalid_argument);
}

TEST(RMSNorm, EmptyThrows) {
    Tensor<float> x;
    Tensor<float> w({4});
    Tensor<float> out;
    EXPECT_THROW(rmsnorm(x, w, out), std::invalid_argument);
    EXPECT_THROW(rmsnorm(x, w), std::invalid_argument);
}
