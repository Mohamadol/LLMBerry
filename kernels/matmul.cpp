#include "llmberry/kernels.h"
#include "llmberry/tensor.h"

#include <stdexcept>

namespace llmberry {

// A [M, K] @ B [K, N] -> C [M, N]
void matmul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& c) {

    ENSURE(a.ndim() == 2 && b.ndim() == 2 && c.ndim() == 2,
           std::invalid_argument,
           "Tensors must be 2D.");
    ENSURE(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] &&
               b.shape()[1] == c.shape()[1],
           std::invalid_argument,
           "Tensors' dimensions mismatch.");

    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];    

    // indexing helpers
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();
    const size_t s1a = a.strides()[0], s2a = a.strides()[1];
    const size_t s1b = b.strides()[0], s2b = b.strides()[1];
    const size_t s1c = c.strides()[0], s2c = c.strides()[1];

    // hot loop
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += A[m * s1a + k * s2a] * B[k * s1b + n * s2b];
            }
            C[m * s1c + n * s2c] = acc;
        }
    }
}

}  // namespace llmberry
