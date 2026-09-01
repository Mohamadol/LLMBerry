#include "llmberry/kernels.h"

#include <cmath>
#include <stdexcept>

namespace llmberry {

void softmax(const Tensor<float>& x, Tensor<float>& out) {
    ENSURE(!x.empty(), std::invalid_argument, "softmax: x cannot be empty");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "softmax: out shape must match x");

    const size_t dim = x.shape().back();
    const size_t n_rows = x.size() / dim;

    // For each vector along the last dimension of `x`:
    //   1. m      = max(x_row)                  // shift for numerical stability
    //   2. e[i]   = exp(x_row[i] - m)           // exp(-inf) == 0
    //   3. out[i] = e[i] / sum(e)
    //
    // Tensor::row(r) is a 1-D view of the r-th last-dim vector.
    // Do not compute exp(x) directly — large logits overflow to inf.

    for (size_t i = 0; i < n_rows; i++) {
        const size_t base_i = i * dim;
        float row_max = x[base_i];
        for (size_t j = 1; j < dim; j++) {
            if (x[base_i + j] > row_max) {
                row_max = x[base_i + j];
            }
        }

        float row_sum = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            const float e = std::exp(x[base_i + j] - row_max);
            out[base_i + j] = e;
            row_sum += e;
        }
        for (size_t j = 0; j < dim; j++) {
            out[base_i + j] /= row_sum;
        }
    }

}

Tensor<float> softmax(const Tensor<float>& x) {
    ENSURE(!x.empty(), std::invalid_argument, "softmax: x cannot be empty");
    Tensor<float> out(x.shape());
    softmax(x, out);
    return out;
}

}  // namespace llmberry
