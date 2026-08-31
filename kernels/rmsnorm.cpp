#include "llmberry/kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llmberry {

void rmsnorm(const Tensor<float>& x,
             const Tensor<float>& weight,
             Tensor<float>& out,
             float eps) {
    ENSURE(!x.empty(), std::invalid_argument, "rmsnorm: x cannot be empty");
    ENSURE(!weight.empty(), std::invalid_argument, "rmsnorm: weight cannot be empty");
    ENSURE(weight.ndim() == 1, std::invalid_argument, "rmsnorm: weight must be 1-D");
    ENSURE(weight.size() == x.shape().back(),
           std::invalid_argument,
           "rmsnorm: weight size must match last dimension of x");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "rmsnorm: out shape must match x");

    const size_t dim = weight.size();
    const size_t n_rows = x.size() / dim;
    // For each vector along the last dimension of `x`:
    //   1. mean_sq  = mean(x_row[i]^2)
    //   2. inv_rms  = 1 / sqrt(mean_sq + eps)   // rsqrt; more stable than x/sqrt
    //   3. out_row[i] = x_row[i] * inv_rms * weight[i]
    //
    // Row-major layout: last dim is contiguous when the last stride is 1.
    // Numerical notes: keep the reduction in float32; always add eps before
    // the square root so an all-zero row does not divide by zero.

    for (size_t i = 0; i < n_rows; ++i) {
        Tensor<float> x_row = x.row(i);
        Tensor<float> out_row = out.row(i);

        float rms = 0.0f;
        for(size_t j=0; j<dim; j++){
            rms += x_row[j]*x_row[j];
        }
        rms /= static_cast<float>(dim);
        float inv_rms = 1 / std::sqrt(rms + eps);

        std::transform(x_row.data(), x_row.data() + dim, weight.data(), out_row.data(),
                       [inv_rms](float xj, float wj) { return xj * inv_rms * wj; });
    }

}

Tensor<float> rmsnorm(const Tensor<float>& x, const Tensor<float>& weight, float eps) {
    ENSURE(!x.empty(), std::invalid_argument, "rmsnorm: x cannot be empty");
    Tensor<float> out(x.shape());
    rmsnorm(x, weight, out, eps);
    return out;
}

}  // namespace llmberry
