#pragma once

#include "llmberry/tensor.h"

namespace llmberry {

/// Llama SwiGLU MLP (Phase 1 checkpoint 08).
///
///   h = SiLU(x @ W_g)  ⊙  (x @ W_u)
///   y = h @ W_d
///
/// No bias (Hugging Face LlamaMLP). Token positions are independent.
///
/// Weight layout is compute-native (not Hugging Face Linear):
///   x:      [..., hidden]
///   w_gate: [hidden, intermediate]
///   w_up:   [hidden, intermediate]
///   w_down: [intermediate, hidden]
///   out:    [..., hidden]   (same shape as x)
///
/// Hugging Face stores Linear weights as [out, in]; transpose once at load
/// (checkpoint 10): W_cpp = W_hf.T.
///
/// Rank: x may be [hidden] (decode), [seq, hidden] (prefill), or
/// [batch, seq, hidden]. Flatten leading dims to [tokens, hidden] and use
/// matmul / silu / mul. `matmul` is 2-D only.

void mlp(const Tensor<float>& x,
         const Tensor<float>& w_gate,
         const Tensor<float>& w_up,
         const Tensor<float>& w_down,
         Tensor<float>& out);
Tensor<float> mlp(const Tensor<float>& x,
                  const Tensor<float>& w_gate,
                  const Tensor<float>& w_up,
                  const Tensor<float>& w_down);

}  // namespace llmberry
