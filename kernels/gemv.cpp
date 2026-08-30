#include "llmberry/kernels.h"

#include <stdexcept>

namespace llmberry {

// A [M, N] @ x [N] -> y [M]
void gemv(const Tensor<float>& a, const Tensor<float>& x, Tensor<float>& y) {
    ENSURE(a.ndim() == 2 && x.ndim() == 1 && y.ndim() == 1,
           std::invalid_argument,
           "a must be 2D, x and y must be 1D.");
    ENSURE(a.shape()[0] == y.shape()[0] && a.shape()[1] == x.shape()[0],
           std::invalid_argument,
           "Tensors' dimensions mismatch.");

    const size_t M = a.shape()[0];
    const size_t N = a.shape()[1];

    const float *A = a.data();
    const float* X = x.data();
    float* Y = y.data();
    const size_t lda = a.strides()[0]; // leading dim of A

    // hot loop
    for (size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                acc += A[m * lda + n] * X[n];
            }
            Y[m] = acc;
    }
}

}  // namespace llmberry
