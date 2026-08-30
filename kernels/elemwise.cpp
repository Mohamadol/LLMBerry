#include "llmberry/kernels.h"

#include <stdexcept>

namespace llmberry {

void add(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) {
    ENSURE(a.shape() == b.shape() && out.shape() == a.shape(),
           std::invalid_argument,
           "add requires matching shapes");

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
}

Tensor<float> add(const Tensor<float>& a, const Tensor<float>& b) {
    ENSURE(a.shape() == b.shape(), std::invalid_argument, "add requires matching shapes");
    Tensor<float> out(a.shape());
    add(a, b, out);
    return out;
}

void mul(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& out) {
    ENSURE(a.shape() == b.shape() && out.shape() == a.shape(),
           std::invalid_argument,
           "mul requires matching shapes");

    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
}

Tensor<float> mul(const Tensor<float>& a, const Tensor<float>& b) {
    ENSURE(a.shape() == b.shape(), std::invalid_argument, "mul requires matching shapes");
    Tensor<float> out(a.shape());
    mul(a, b, out);
    return out;
}

}  // namespace llmberry
