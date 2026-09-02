#include "llmberry/mlp.h"
#include "llmberry/ensure.h"
#include "llmberry/kernels.h"

#include <stdexcept>

namespace llmberry {
namespace {

void check_mlp(const Tensor<float>& x,
               const Tensor<float>& w_gate,
               const Tensor<float>& w_up,
               const Tensor<float>& w_down,
               const Tensor<float>& out) {
    ENSURE(!x.empty() && !w_gate.empty() && !w_up.empty() && !w_down.empty(),
           std::invalid_argument,
           "mlp: x and weights cannot be empty");
    ENSURE(x.ndim() >= 1, std::invalid_argument, "mlp: x must have shape [..., hidden]");
    ENSURE(w_gate.ndim() == 2 && w_up.ndim() == 2 && w_down.ndim() == 2,
           std::invalid_argument,
           "mlp: weights must be 2-D");

    const size_t hidden = w_gate.shape()[0];
    const size_t intermediate = w_gate.shape()[1];
    ENSURE(x.shape().back() == hidden, std::invalid_argument,
           "mlp: x last dim must match w_gate.shape()[0] (hidden)");
    ENSURE(w_up.shape() == w_gate.shape(), std::invalid_argument,
           "mlp: w_up must have the same shape as w_gate [hidden, intermediate]");
    ENSURE(w_down.shape()[0] == intermediate && w_down.shape()[1] == hidden,
           std::invalid_argument,
           "mlp: w_down must have shape [intermediate, hidden]");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "mlp: out shape must match x");
}

}  // namespace

void mlp(const Tensor<float>& x,
         const Tensor<float>& w_gate,
         const Tensor<float>& w_up,
         const Tensor<float>& w_down,
         Tensor<float>& out) {
    check_mlp(x, w_gate, w_up, w_down, out);

    const size_t hidden = w_gate.shape()[0];
    const size_t intermediate = w_gate.shape()[1];
    const size_t tokens = x.size() / hidden;

    Tensor<float> x2d = x.view({tokens, hidden});
    Tensor<float> out2d = out.view({tokens, hidden});

    Tensor<float> gate({tokens, intermediate});
    Tensor<float> up({tokens, intermediate});
    Tensor<float> h({tokens, intermediate});

    matmul(x2d, w_gate, gate);
    matmul(x2d, w_up, up);
    silu(gate, gate);
    mul(gate, up, h);
    matmul(h, w_down, out2d);
}

Tensor<float> mlp(const Tensor<float>& x,
                  const Tensor<float>& w_gate,
                  const Tensor<float>& w_up,
                  const Tensor<float>& w_down) {
    ENSURE(!x.empty(), std::invalid_argument, "mlp: x cannot be empty");
    Tensor<float> out(x.shape());
    mlp(x, w_gate, w_up, w_down, out);
    return out;
}

}  // namespace llmberry
