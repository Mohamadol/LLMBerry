#include "llmberry/mlp.h"
#include "llmberry/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using llmberry::mlp;
using llmberry::Tensor;

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

float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

/// Independent SwiGLU: per-token dots, not llmberry::matmul / mlp.
void mlp_ref(const Tensor<float>& x,
             const Tensor<float>& w_gate,
             const Tensor<float>& w_up,
             const Tensor<float>& w_down,
             Tensor<float>& out) {
    const size_t hidden = w_gate.shape()[0];
    const size_t intermediate = w_gate.shape()[1];
    ASSERT_EQ(x.size() % hidden, 0u);
    const size_t tokens = x.size() / hidden;

    for (size_t t = 0; t < tokens; ++t) {
        std::vector<float> hidden_act(intermediate);
        for (size_t i = 0; i < intermediate; ++i) {
            float gate = 0.0f;
            float up = 0.0f;
            for (size_t k = 0; k < hidden; ++k) {
                const float xk = x[t * hidden + k];
                gate += xk * w_gate.at({k, i});
                up += xk * w_up.at({k, i});
            }
            hidden_act[i] = silu_ref(gate) * up;
        }
        for (size_t j = 0; j < hidden; ++j) {
            float y = 0.0f;
            for (size_t i = 0; i < intermediate; ++i) {
                y += hidden_act[i] * w_down.at({i, j});
            }
            out[t * hidden + j] = y;
        }
    }
}

void run_mlp_vs_ref(const std::vector<size_t>& x_shape, size_t intermediate, float x_start) {
    const size_t hidden = x_shape.back();
    Tensor<float> x(x_shape);
    Tensor<float> w_gate({hidden, intermediate});
    Tensor<float> w_up({hidden, intermediate});
    Tensor<float> w_down({intermediate, hidden});
    Tensor<float> out(x_shape);
    Tensor<float> want(x_shape);
    fill_iota(x, x_start);
    fill_pattern(w_gate);
    fill_iota(w_up, -0.5f);
    fill_pattern(w_down);

    mlp(x, w_gate, w_up, w_down, out);
    mlp_ref(x, w_gate, w_up, w_down, want);
    expect_allclose(out, want);
}

}  // namespace

TEST(MLP, KnownValuesTiny) {
    // x = [1, 0]
    // W_g = [[1, 0, 0],
    //        [0, 1, 0]]          [H=2, I=3]  →  gate = [1, 0, 0]
    // W_u = ones                 →  up   = [1, 1, 1]
    // h = [silu(1), 0, 0]
    // W_d = [[1, 0],
    //        [0, 1],
    //        [0, 0]]             [I=3, H=2]  →  y = [silu(1), 0]
    Tensor<float> x({2});
    x[0] = 1.0f;
    x[1] = 0.0f;

    Tensor<float> w_gate({2, 3});
    Tensor<float> w_up = Tensor<float>::ones({2, 3});
    Tensor<float> w_down({3, 2});
    Tensor<float> out({2});
    for (size_t i = 0; i < w_gate.size(); ++i) {
        w_gate[i] = 0.0f;
        w_down[i] = 0.0f;
    }
    w_gate.at({0, 0}) = 1.0f;
    w_gate.at({1, 1}) = 1.0f;
    w_down.at({0, 0}) = 1.0f;
    w_down.at({1, 1}) = 1.0f;

    mlp(x, w_gate, w_up, w_down, out);

    EXPECT_NEAR(out[0], silu_ref(1.0f), kTol);
    EXPECT_NEAR(out[1], 0.0f, kTol);
}

TEST(MLP, ZeroGateIsZero) {
    Tensor<float> x({3, 4});
    Tensor<float> w_gate = Tensor<float>::zeros({4, 6});
    Tensor<float> w_up({4, 6});
    Tensor<float> w_down({6, 4});
    Tensor<float> out({3, 4});
    fill_iota(x);
    fill_pattern(w_up);
    fill_iota(w_down, -1.0f);
    fill_iota(out, 99.0f);

    mlp(x, w_gate, w_up, w_down, out);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 0.0f, kTol) << "index " << i;
    }
}

TEST(MLP, ZeroDownIsZero) {
    Tensor<float> x({2, 4});
    Tensor<float> w_gate({4, 5});
    Tensor<float> w_up({4, 5});
    Tensor<float> w_down = Tensor<float>::zeros({5, 4});
    Tensor<float> out({2, 4});
    fill_pattern(x);
    fill_iota(w_gate);
    fill_pattern(w_up);
    fill_iota(out, 99.0f);

    mlp(x, w_gate, w_up, w_down, out);

    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], 0.0f, kTol) << "index " << i;
    }
}

TEST(MLP, PrefillEqualsPerTokenDecode) {
    Tensor<float> x({3, 4});
    Tensor<float> w_gate({4, 6});
    Tensor<float> w_up({4, 6});
    Tensor<float> w_down({6, 4});
    Tensor<float> out({3, 4});
    fill_iota(x, -2.0f);
    fill_pattern(w_gate);
    fill_iota(w_up, 0.25f);
    fill_pattern(w_down);

    mlp(x, w_gate, w_up, w_down, out);

    for (size_t t = 0; t < 3; ++t) {
        Tensor<float> row_out({4});
        mlp(x.row(t), w_gate, w_up, w_down, row_out);
        Tensor<float> want_row = out.row(t);
        expect_allclose(row_out, want_row);
    }
}

TEST(MLP, MatchesIndependentReference) {
    run_mlp_vs_ref({4}, 6, -2.0f);
    run_mlp_vs_ref({3, 4}, 6, 1.0f);
    run_mlp_vs_ref({2, 3, 4}, 11, -1.0f);
}

TEST(MLP, OverwritesOut) {
    Tensor<float> x({2, 4});
    Tensor<float> w_gate({4, 6});
    Tensor<float> w_up({4, 6});
    Tensor<float> w_down({6, 4});
    Tensor<float> out({2, 4});
    Tensor<float> want({2, 4});
    fill_iota(x);
    fill_pattern(w_gate);
    fill_iota(w_up, -0.5f);
    fill_pattern(w_down);
    fill_iota(out, 100.0f);

    mlp(x, w_gate, w_up, w_down, out);
    mlp_ref(x, w_gate, w_up, w_down, want);
    expect_allclose(out, want);
}

TEST(MLP, LeavesInputsUnchanged) {
    Tensor<float> x({2, 4});
    Tensor<float> w_gate({4, 5});
    Tensor<float> w_up({4, 5});
    Tensor<float> w_down({5, 4});
    Tensor<float> x_copy({2, 4});
    Tensor<float> gate_copy({4, 5});
    Tensor<float> up_copy({4, 5});
    Tensor<float> down_copy({5, 4});
    Tensor<float> out({2, 4});
    fill_pattern(x);
    fill_iota(w_gate);
    fill_pattern(w_up);
    fill_iota(w_down, -1.0f);
    copy_tensor(x, x_copy);
    copy_tensor(w_gate, gate_copy);
    copy_tensor(w_up, up_copy);
    copy_tensor(w_down, down_copy);

    mlp(x, w_gate, w_up, w_down, out);

    expect_allclose(x, x_copy);
    expect_allclose(w_gate, gate_copy);
    expect_allclose(w_up, up_copy);
    expect_allclose(w_down, down_copy);
}

TEST(MLP, AllocatingWrapper) {
    Tensor<float> x({2, 3});
    Tensor<float> w_gate({3, 5});
    Tensor<float> w_up({3, 5});
    Tensor<float> w_down({5, 3});
    Tensor<float> want({2, 3});
    fill_iota(x, -1.0f);
    fill_pattern(w_gate);
    fill_iota(w_up, 0.5f);
    fill_pattern(w_down);

    Tensor<float> out = mlp(x, w_gate, w_up, w_down);
    EXPECT_EQ(out.shape(), x.shape());
    mlp_ref(x, w_gate, w_up, w_down, want);
    expect_allclose(out, want);
}

TEST(MLP, ContiguousReshapeView) {
    Tensor<float> storage({24});
    fill_iota(storage, 1.0f);
    Tensor<float> x = storage.view({2, 3, 4});
    Tensor<float> w_gate({4, 7});
    Tensor<float> w_up({4, 7});
    Tensor<float> w_down({7, 4});
    Tensor<float> out({2, 3, 4});
    Tensor<float> want({2, 3, 4});
    fill_pattern(w_gate);
    fill_iota(w_up, -0.25f);
    fill_pattern(w_down);

    mlp(x, w_gate, w_up, w_down, out);
    mlp_ref(x, w_gate, w_up, w_down, want);
    expect_allclose(out, want);
}

TEST(MLP, ShapeMismatchThrows) {
    Tensor<float> x({2, 4});
    Tensor<float> w_gate({4, 6});
    Tensor<float> w_up({4, 6});
    Tensor<float> w_down({6, 4});
    Tensor<float> out({2, 4});

    Tensor<float> out_bad({4, 2});
    Tensor<float> gate_hidden({5, 6});
    Tensor<float> up_mismatch({4, 5});
    Tensor<float> down_i({5, 4});
    Tensor<float> down_h({6, 3});
    Tensor<float> gate_1d({4});
    Tensor<float> down_3d({6, 4, 1});

    EXPECT_THROW(mlp(x, w_gate, w_up, w_down, out_bad), std::invalid_argument);
    EXPECT_THROW(mlp(x, gate_hidden, w_up, w_down, out), std::invalid_argument);
    EXPECT_THROW(mlp(x, w_gate, up_mismatch, w_down, out), std::invalid_argument);
    EXPECT_THROW(mlp(x, w_gate, w_up, down_i, out), std::invalid_argument);
    EXPECT_THROW(mlp(x, w_gate, w_up, down_h, out), std::invalid_argument);
    EXPECT_THROW(mlp(x, gate_1d, w_up, w_down, out), std::invalid_argument);
    EXPECT_THROW(mlp(x, w_gate, w_up, down_3d, out), std::invalid_argument);
}

TEST(MLP, EmptyThrows) {
    Tensor<float> x;
    Tensor<float> w_gate({4, 6});
    Tensor<float> w_up({4, 6});
    Tensor<float> w_down({6, 4});
    Tensor<float> out;
    EXPECT_THROW(mlp(x, w_gate, w_up, w_down, out), std::invalid_argument);
    EXPECT_THROW(mlp(x, w_gate, w_up, w_down), std::invalid_argument);
}
