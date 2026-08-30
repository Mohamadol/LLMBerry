#include "llmberry/ensure.h"
#include "llmberry/kernels.h"

#include <stdexcept>

namespace llmberry {

float dot(const Tensor<float>& a, const Tensor<float>& b) {
    ENSURE(a.ndim()==1 && b.ndim()==1, std::invalid_argument, "Tensors must be 1D");
    ENSURE(a.size()==b.size(), std::invalid_argument, "Tensors must be the same size");
    ENSURE(a.size()!=0, std::invalid_argument, "Tensors cannot be empty");

    float acc = 0.0f;
    for(size_t i=0; i<a.size();i++){
        acc += a[i] * b[i];
    }
    return acc;
}

float reduce_sum(const Tensor<float>& x) {
    ENSURE(x.size()!=0, std::invalid_argument, "Tensor cannot be empty");

    float acc = 0.0f;
    for(size_t i=0; i<x.size();i++){
        acc += x[i];
    }
    return acc;
}

float reduce_mean(const Tensor<float>& x) {
    ENSURE(x.size()!=0, std::invalid_argument, "Tensor cannot be empty");
    return reduce_sum(x) / static_cast<float>(x.size());
}

}  // namespace llmberry
