#include "llmberry/kernels.h"

#include <cmath>
#include <stdexcept>

namespace llmberry {

void gelu(const Tensor<float>& x, Tensor<float>& out) {
    ENSURE(!x.empty(), std::invalid_argument, "tensors cannot be empty");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "gelu requires matching shapes");

    constexpr float inv_sqrt2 = 0.7071067811865476f;  // 1 / sqrt(2)
    for (size_t i = 0; i < x.size(); ++i) {
        const float xi = x[i];
        out[i] = 0.5f * xi * (1.0f + std::erf(xi * inv_sqrt2));
    }
}

Tensor<float> gelu(const Tensor<float>& x) {
    ENSURE(!x.empty(), std::invalid_argument, "tensors cannot be empty");
    Tensor<float> out(x.shape());
    gelu(x, out);
    return out;
}

}  // namespace llmberry
