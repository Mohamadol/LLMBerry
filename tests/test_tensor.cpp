#include "llmberry/tensor.h"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace {

using llmberry::DType;
using llmberry::Tensor;
using llmberry::dtype_of;

std::vector<size_t> expected_row_major_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

size_t tensor_size(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

}  // namespace

// --- Construction & metadata ---

TEST(TensorConstruction, ShapeAndStridesFor2x4x8) {
    Tensor<float> x({2, 4, 8});

    EXPECT_EQ(x.ndim(), 3);
    EXPECT_EQ(x.shape(), std::vector<size_t>({2, 4, 8}));
    EXPECT_EQ(x.strides(), expected_row_major_strides({2, 4, 8}));
    EXPECT_EQ(x.size(), 64);
    EXPECT_EQ(x.nbytes(), 64 * sizeof(float));
    EXPECT_TRUE(x.owns_data());
    EXPECT_FALSE(x.is_view());
}

TEST(TensorConstruction, ShapeAndStridesFor1D) {
    Tensor<int32_t> x({10});

    EXPECT_EQ(x.ndim(), 1);
    EXPECT_EQ(x.shape(), std::vector<size_t>({10}));
    EXPECT_EQ(x.strides(), std::vector<size_t>({1}));
    EXPECT_EQ(x.size(), 10);
    EXPECT_EQ(x.dtype(), DType::Int32);
}

TEST(TensorConstruction, RejectsZeroSizedDimension) {
    EXPECT_THROW(Tensor<float>({2, 0, 8}), std::invalid_argument);
}

// --- Dtype ---

TEST(TensorDtype, MapsElementTypes) {
    EXPECT_EQ(dtype_of<float>(), DType::Float32);
    EXPECT_EQ(dtype_of<double>(), DType::Float64);
    EXPECT_EQ(dtype_of<int32_t>(), DType::Int32);
    EXPECT_EQ(dtype_of<int64_t>(), DType::Int64);

    Tensor<float> f({2, 2});
    Tensor<double> d({2, 2});
    Tensor<int32_t> i({2, 2});
    Tensor<int64_t> l({2, 2});

    EXPECT_EQ(f.dtype(), DType::Float32);
    EXPECT_EQ(d.dtype(), DType::Float64);
    EXPECT_EQ(i.dtype(), DType::Int32);
    EXPECT_EQ(l.dtype(), DType::Int64);
}

// --- zeros / ones ---

TEST(TensorFactories, Zeros) {
    Tensor<float> x = Tensor<float>::zeros({3, 4});

    EXPECT_EQ(x.size(), 12);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_FLOAT_EQ(x[i], 0.0f);
    }
}

TEST(TensorFactories, Ones) {
    Tensor<float> x = Tensor<float>::ones({2, 3});

    EXPECT_EQ(x.size(), 6);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_FLOAT_EQ(x[i], 1.0f);
    }
}

// --- Flat indexing ---

TEST(TensorIndexing, FlatAccess) {
    Tensor<float> x({2, 3});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i);
    }

    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_FLOAT_EQ(x[i], static_cast<float>(i));
    }
}

TEST(TensorIndexing, FlatAccessOutOfBounds) {
    Tensor<float> x({2, 3});
    EXPECT_THROW(x[6], std::out_of_range);
}

// --- Multi-dimensional indexing ---

TEST(TensorIndexing, At2D) {
    Tensor<float> x({2, 3});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i);
    }

    EXPECT_FLOAT_EQ(x.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(x.at({0, 2}), 2.0f);
    EXPECT_FLOAT_EQ(x.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(x.at({1, 2}), 5.0f);
}

TEST(TensorIndexing, At3D) {
    Tensor<float> x({2, 2, 2});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i);
    }

    EXPECT_FLOAT_EQ(x.at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(x.at({0, 1, 1}), 3.0f);
    EXPECT_FLOAT_EQ(x.at({1, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(x.at({1, 1, 1}), 7.0f);
}

TEST(TensorIndexing, AtWrongRankThrows) {
    Tensor<float> x({2, 3});
    EXPECT_THROW(x.at({0}), std::out_of_range);
    EXPECT_THROW(x.at({0, 0, 0}), std::out_of_range);
}

TEST(TensorIndexing, AtOutOfBoundsThrows) {
    Tensor<float> x({2, 3});
    EXPECT_THROW(x.at({2, 0}), std::out_of_range);
    EXPECT_THROW(x.at({0, 3}), std::out_of_range);
}

// --- data() pointer ---

TEST(TensorData, DataPointerMatchesIndexedValues) {
    Tensor<float> x({2, 2});
    x.at({0, 0}) = 1.0f;
    x.at({0, 1}) = 2.0f;
    x.at({1, 0}) = 3.0f;
    x.at({1, 1}) = 4.0f;

    const float* ptr = x.data();
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(ptr[2], 3.0f);
    EXPECT_FLOAT_EQ(ptr[3], 4.0f);
}

// --- Views ---

TEST(TensorView, ReshapeSharesStorage) {
    Tensor<float> x({2, 6});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i);
    }

    Tensor<float> v = x.view({3, 4});
    EXPECT_EQ(v.shape(), std::vector<size_t>({3, 4}));
    EXPECT_EQ(v.size(), 12);
    EXPECT_FALSE(v.owns_data());
    EXPECT_TRUE(v.is_view());

    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
    }

    v.at({2, 3}) = 99.0f;
    EXPECT_FLOAT_EQ(x.at({1, 5}), 99.0f);
}

TEST(TensorView, RowSliceSharesStorage) {
    Tensor<float> x({2, 3});
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i + 1);
    }

    Tensor<float> r0 = x.row(0);
    Tensor<float> r1 = x.row(1);

    EXPECT_EQ(r0.shape(), std::vector<size_t>({3}));
    EXPECT_TRUE(r0.is_view());
    EXPECT_FLOAT_EQ(r0[0], 1.0f);
    EXPECT_FLOAT_EQ(r0[2], 3.0f);
    EXPECT_FLOAT_EQ(r1[0], 4.0f);
    EXPECT_FLOAT_EQ(r1[2], 6.0f);

    r1[1] = 99.0f;
    EXPECT_FLOAT_EQ(x.at({1, 1}), 99.0f);
}

TEST(TensorView, RowOutOfRangeThrows) {
    Tensor<float> x({2, 3});
    EXPECT_THROW(x.row(2), std::out_of_range);
}

TEST(TensorView, InvalidReshapeThrows) {
    Tensor<float> x({2, 6});
    EXPECT_THROW(x.view({2, 5}), std::invalid_argument);
}

TEST(TensorView, ExternalBufferView) {
    std::vector<float> buffer(6, 0.0f);
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<float>(i + 1);
    }

    Tensor<float> x({2, 3}, {3, 1}, buffer.data());
    EXPECT_FALSE(x.owns_data());
    EXPECT_TRUE(x.is_view());

    EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(x.at({1, 2}), 6.0f);
}

// --- Row-major layout sanity check ---

TEST(TensorLayout, RowMajorLinearization) {
    Tensor<float> x({2, 3});
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            x.at({row, col}) = static_cast<float>(row * 3 + col);
        }
    }

    const float* ptr = x.data();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i));
    }
}
