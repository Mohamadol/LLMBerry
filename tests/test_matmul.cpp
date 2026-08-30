#include "llmberry/kernels.h"
#include "llmberry/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using llmberry::Tensor;
using llmberry::add;
using llmberry::dot;
using llmberry::gemv;
using llmberry::matmul;
using llmberry::mul;
using llmberry::reduce_mean;
using llmberry::reduce_sum;
using llmberry::gelu;
using llmberry::silu;

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

void matmul_ref(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {
    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += a.at({m, k}) * b.at({k, n});
            }
            c.at({m, n}) = acc;
        }
    }
}

void expect_allclose(const Tensor<float>& got, const Tensor<float>& want, float tol = 1e-5f) {
    ASSERT_EQ(got.shape(), want.shape());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
    }
}

float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

float gelu_ref(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

void expect_unary_matches_ref(const Tensor<float>& got,
                              const Tensor<float>& x,
                              float (*ref)(float),
                              float tol = 1e-5f) {
    ASSERT_EQ(got.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(got[i], ref(x[i]), tol) << "index " << i;
    }
}

}  // namespace

// --- matmul ---

TEST(Matmul, SmallKnownValues) {
    Tensor<float> a({2, 3});
    Tensor<float> b({3, 2});
    Tensor<float> c({2, 2});
    fill_iota(a, 1.0f);
    fill_iota(b, 1.0f);

    matmul(a, b, c);

    // A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
    // C = [[22, 28], [49, 64]]
    EXPECT_FLOAT_EQ(c.at({0, 0}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 28.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 49.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 64.0f);
}

TEST(Matmul, OneByOne) {
    Tensor<float> a({1, 1});
    Tensor<float> b({1, 1});
    Tensor<float> c({1, 1});
    a[0] = 2.5f;
    b[0] = -4.0f;
    c[0] = 99.0f;

    matmul(a, b, c);
    EXPECT_FLOAT_EQ(c[0], -10.0f);
}

TEST(Matmul, IdentityLeftAndRight) {
    Tensor<float> a({3, 3});
    Tensor<float> i({3, 3});
    Tensor<float> c({3, 3});
    fill_iota(a, 1.0f);
    for (size_t r = 0; r < 3; ++r) {
        for (size_t col = 0; col < 3; ++col) {
            i.at({r, col}) = (r == col) ? 1.0f : 0.0f;
        }
    }

    matmul(i, a, c);
    expect_allclose(c, a);

    matmul(a, i, c);
    expect_allclose(c, a);
}

TEST(Matmul, OuterProduct) {
    Tensor<float> a({3, 1});
    Tensor<float> b({1, 2});
    Tensor<float> c({3, 2});
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    b[0] = 4.0f;
    b[1] = 5.0f;

    matmul(a, b, c);

    EXPECT_FLOAT_EQ(c.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 8.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 10.0f);
    EXPECT_FLOAT_EQ(c.at({2, 0}), 12.0f);
    EXPECT_FLOAT_EQ(c.at({2, 1}), 15.0f);
}

TEST(Matmul, TimesZeroIsZero) {
    Tensor<float> a({4, 3});
    Tensor<float> z = Tensor<float>::zeros({3, 5});
    Tensor<float> c({4, 5});
    fill_pattern(a);
    fill_iota(c, 7.0f);

    matmul(a, z, c);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c[i], 0.0f);
    }
}

TEST(Matmul, OverwritesOutput) {
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    Tensor<float> c({2, 2});
    fill_iota(a, 1.0f);
    fill_iota(b, 1.0f);
    fill_iota(c, 100.0f);

    matmul(a, b, c);

    EXPECT_FLOAT_EQ(c.at({0, 0}), 7.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 10.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 15.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 22.0f);
}

TEST(Matmul, LeavesInputsUnchanged) {
    Tensor<float> a({3, 4});
    Tensor<float> b({4, 2});
    Tensor<float> a_copy({3, 4});
    Tensor<float> b_copy({4, 2});
    Tensor<float> c({3, 2});
    fill_pattern(a);
    fill_iota(b, -2.0f);
    copy_tensor(a, a_copy);
    copy_tensor(b, b_copy);

    matmul(a, b, c);

    expect_allclose(a, a_copy);
    expect_allclose(b, b_copy);
}

TEST(Matmul, MatchesIndependentReference) {
    const std::vector<std::vector<size_t>> cases = {
        {1, 1, 1},
        {1, 8, 1},
        {5, 1, 4},
        {2, 3, 2},
        {4, 7, 3},
        {8, 5, 6},
    };

    for (const auto& dims : cases) {
        const size_t M = dims[0];
        const size_t K = dims[1];
        const size_t N = dims[2];
        Tensor<float> a({M, K});
        Tensor<float> b({K, N});
        Tensor<float> got({M, N});
        Tensor<float> want({M, N});
        fill_pattern(a);
        fill_iota(b, -1.5f);

        matmul(a, b, got);
        matmul_ref(a, b, want);
        expect_allclose(got, want);
    }
}

TEST(Matmul, ContiguousReshapeView) {
    Tensor<float> storage({12});
    fill_iota(storage, 1.0f);
    Tensor<float> a = storage.view({3, 4});
    Tensor<float> b({4, 2});
    Tensor<float> c({3, 2});
    Tensor<float> want({3, 2});
    fill_iota(b, 2.0f);

    matmul(a, b, c);
    matmul_ref(a, b, want);
    expect_allclose(c, want);
}

TEST(Matmul, RejectsNon2D) {
    Tensor<float> a2({2, 3});
    Tensor<float> b2({3, 2});
    Tensor<float> c2({2, 2});
    Tensor<float> a1({6});
    Tensor<float> b1({6});
    Tensor<float> c1({4});
    Tensor<float> a3({2, 3, 1});
    Tensor<float> b3({3, 2, 1});
    Tensor<float> c3({2, 2, 1});

    EXPECT_THROW(matmul(a1, b2, c2), std::invalid_argument);
    EXPECT_THROW(matmul(a2, b1, c2), std::invalid_argument);
    EXPECT_THROW(matmul(a2, b2, c1), std::invalid_argument);
    EXPECT_THROW(matmul(a3, b2, c2), std::invalid_argument);
    EXPECT_THROW(matmul(a2, b3, c2), std::invalid_argument);
    EXPECT_THROW(matmul(a2, b2, c3), std::invalid_argument);
}

TEST(Matmul, ShapeMismatchThrows) {
    Tensor<float> a({2, 3});
    Tensor<float> b({3, 2});
    Tensor<float> c({2, 2});

    Tensor<float> b_bad_k({4, 2});
    Tensor<float> c_bad_m({3, 2});
    Tensor<float> c_bad_n({2, 3});

    EXPECT_THROW(matmul(a, b_bad_k, c), std::invalid_argument);
    EXPECT_THROW(matmul(a, b, c_bad_m), std::invalid_argument);
    EXPECT_THROW(matmul(a, b, c_bad_n), std::invalid_argument);
}

// --- gemv ---

TEST(Gemv, SmallKnownValues) {
    Tensor<float> a({2, 3});
    Tensor<float> x({3});
    Tensor<float> y({2});
    fill_iota(a, 1.0f);
    fill_iota(x, 1.0f);

    gemv(a, x, y);

    // A @ [1,2,3] = [14, 32]
    EXPECT_FLOAT_EQ(y[0], 14.0f);
    EXPECT_FLOAT_EQ(y[1], 32.0f);
}

TEST(Gemv, ShapeMismatchThrows) {
    Tensor<float> a({2, 3});
    Tensor<float> x({3});
    Tensor<float> y({2});
    Tensor<float> x_bad({4});
    Tensor<float> y_bad({3});

    EXPECT_THROW(gemv(a, x_bad, y), std::invalid_argument);
    EXPECT_THROW(gemv(a, x, y_bad), std::invalid_argument);
}

// --- elementwise ---

TEST(Elemwise, Add) {
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    Tensor<float> out({2, 2});
    fill_iota(a, 1.0f);
    fill_iota(b, 10.0f);

    add(a, b, out);

    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 13.0f);
    EXPECT_FLOAT_EQ(out[2], 15.0f);
    EXPECT_FLOAT_EQ(out[3], 17.0f);
}

TEST(Elemwise, Mul) {
    Tensor<float> a({4});
    Tensor<float> b({4});
    Tensor<float> out({4});
    fill_iota(a, 1.0f);
    fill_iota(b, 2.0f);

    mul(a, b, out);

    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 6.0f);
    EXPECT_FLOAT_EQ(out[2], 12.0f);
    EXPECT_FLOAT_EQ(out[3], 20.0f);
}

TEST(Elemwise, AllocatingWrappers) {
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    fill_iota(a, 1.0f);
    fill_iota(b, 10.0f);

    Tensor<float> sum = add(a, b);
    Tensor<float> prod = mul(a, b);

    EXPECT_EQ(sum.shape(), a.shape());
    EXPECT_EQ(prod.shape(), a.shape());
    EXPECT_FLOAT_EQ(sum[0], 11.0f);
    EXPECT_FLOAT_EQ(prod[0], 10.0f);
    EXPECT_FLOAT_EQ(prod[3], 52.0f);
}

TEST(Elemwise, ShapeMismatchThrows) {
    Tensor<float> a({2, 3});
    Tensor<float> b({6});
    Tensor<float> out({2, 3});
    Tensor<float> out_wrong({3, 2});

    EXPECT_THROW(add(a, b, out), std::invalid_argument);
    EXPECT_THROW(mul(a, b, out), std::invalid_argument);
    EXPECT_THROW(add(a, a, out_wrong), std::invalid_argument);
    EXPECT_THROW(add(a, b), std::invalid_argument);
    EXPECT_THROW(mul(a, b), std::invalid_argument);
}

// --- dot / reductions ---

TEST(Dot, EqualLengthVectors) {
    Tensor<float> a({3});
    Tensor<float> b({3});
    fill_iota(a, 1.0f);
    fill_iota(b, 2.0f);

    EXPECT_FLOAT_EQ(dot(a, b), 1.0f * 2.0f + 2.0f * 3.0f + 3.0f * 4.0f);
}

TEST(Reduce, SumAndMean) {
    Tensor<float> x({2, 2});
    fill_iota(x, 1.0f);

    EXPECT_FLOAT_EQ(reduce_sum(x), 10.0f);
    EXPECT_FLOAT_EQ(reduce_mean(x), 2.5f);
}

// --- silu ---

TEST(Silu, ZeroIsZero) {
    Tensor<float> x({3});
    Tensor<float> out({3});
    x[0] = 0.0f;
    x[1] = 1.0f;
    x[2] = -1.0f;

    silu(x, out);

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_NEAR(out[1], silu_ref(1.0f), 1e-6f);
    EXPECT_NEAR(out[2], silu_ref(-1.0f), 1e-6f);
}

TEST(Silu, MatchesReference) {
    Tensor<float> x({2, 4});
    Tensor<float> out({2, 4});
    fill_pattern(x);
    x[0] = 8.0f;
    x[7] = -8.0f;

    silu(x, out);
    expect_unary_matches_ref(out, x, silu_ref);
}

TEST(Silu, OverwritesOutput) {
    Tensor<float> x({4});
    Tensor<float> out({4});
    fill_iota(x, -1.0f);
    fill_iota(out, 100.0f);

    silu(x, out);
    expect_unary_matches_ref(out, x, silu_ref);
}

TEST(Silu, LeavesInputUnchanged) {
    Tensor<float> x({5});
    Tensor<float> x_copy({5});
    Tensor<float> out({5});
    fill_pattern(x);
    copy_tensor(x, x_copy);

    silu(x, out);
    expect_allclose(x, x_copy);
}

TEST(Silu, AllocatingWrapper) {
    Tensor<float> x({2, 2});
    fill_iota(x, -0.5f);

    Tensor<float> out = silu(x);
    EXPECT_EQ(out.shape(), x.shape());
    expect_unary_matches_ref(out, x, silu_ref);
}

TEST(Silu, ShapeMismatchThrows) {
    Tensor<float> x({2, 3});
    Tensor<float> out({6});
    EXPECT_THROW(silu(x, out), std::invalid_argument);
}

TEST(Silu, EmptyThrows) {
    Tensor<float> x;
    Tensor<float> out;
    EXPECT_THROW(silu(x, out), std::invalid_argument);
    EXPECT_THROW(silu(x), std::invalid_argument);
}

// --- gelu ---

TEST(Gelu, ZeroIsZero) {
    Tensor<float> x({1});
    Tensor<float> out({1});
    x[0] = 0.0f;

    gelu(x, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(Gelu, MatchesReference) {
    Tensor<float> x({2, 4});
    Tensor<float> out({2, 4});
    fill_pattern(x);
    x[0] = 1.0f;
    x[1] = -1.0f;
    x[2] = 3.0f;

    gelu(x, out);
    expect_unary_matches_ref(out, x, gelu_ref, 1e-5f);
}

TEST(Gelu, OverwritesOutput) {
    Tensor<float> x({3});
    Tensor<float> out({3});
    fill_iota(x, -1.0f);
    fill_iota(out, 50.0f);

    gelu(x, out);
    expect_unary_matches_ref(out, x, gelu_ref);
}

TEST(Gelu, AllocatingWrapper) {
    Tensor<float> x({3});
    fill_iota(x, -1.0f);

    Tensor<float> out = gelu(x);
    EXPECT_EQ(out.shape(), x.shape());
    expect_unary_matches_ref(out, x, gelu_ref);
}

TEST(Gelu, ShapeMismatchThrows) {
    Tensor<float> x({2, 3});
    Tensor<float> out({3, 2});
    EXPECT_THROW(gelu(x, out), std::invalid_argument);
}

TEST(Gelu, EmptyThrows) {
    Tensor<float> x;
    Tensor<float> out;
    EXPECT_THROW(gelu(x, out), std::invalid_argument);
    EXPECT_THROW(gelu(x), std::invalid_argument);
}
