#include "llmberry/kernels.h"

#include <stdexcept>
#include <vector>

namespace llmberry {

void rope_freqs(size_t dim, Tensor<float>& inv_freq, float theta) {
    ENSURE(dim > 0, std::invalid_argument, "rope_freqs: dim must be positive");
    ENSURE(dim % 2 == 0, std::invalid_argument, "rope_freqs: dim must be even");
    ENSURE(theta > 0.0f, std::invalid_argument, "rope_freqs: theta must be positive");
    ENSURE(inv_freq.ndim() == 1 && inv_freq.size() == dim / 2,
           std::invalid_argument,
           "rope_freqs: inv_freq must have shape [dim/2]");

    // inv_freq[i] = 1 / theta^(2i / dim)    for i = 0 .. dim/2 - 1
    // Equivalent:  theta ** (-(2i) / dim)
    // Hugging Face: 1.0 / (base ** (arange(0, dim, 2) / dim))

    for(size_t i=0; i<dim/2; i++){
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(dim));
    }
}

Tensor<float> rope_freqs(size_t dim, float theta) {
    ENSURE(dim > 0, std::invalid_argument, "rope_freqs: dim must be positive");
    ENSURE(dim % 2 == 0, std::invalid_argument, "rope_freqs: dim must be even");
    Tensor<float> inv_freq({dim / 2});
    rope_freqs(dim, inv_freq, theta);
    return inv_freq;
}

void rope_sincos(const Tensor<float>& inv_freq,
                 const std::vector<size_t>& positions,
                 Tensor<float>& cos,
                 Tensor<float>& sin) {
    ENSURE(!inv_freq.empty() && inv_freq.ndim() == 1,
           std::invalid_argument,
           "rope_sincos: inv_freq must be 1-D and non-empty");
    ENSURE(!positions.empty(), std::invalid_argument, "rope_sincos: positions cannot be empty");

    const size_t half = inv_freq.size();
    const size_t dim = half * 2;
    const size_t n_pos = positions.size();
    const std::vector<size_t> table_shape = {n_pos, dim};
    ENSURE(cos.shape() == table_shape && sin.shape() == table_shape,
           std::invalid_argument,
           "rope_sincos: cos and sin must have shape [n_pos, dim]");

    // For each position t and pair index i:
    //   angle = positions[t] * inv_freq[i]
    //   cos[t, i] = cos[t, i + half] = cos(angle)
    //   sin[t, i] = sin[t, i + half] = sin(angle)
    //
    // That concat(freqs, freqs) layout matches Hugging Face LlamaRotaryEmbedding.

    for (size_t t = 0; t < n_pos; ++t){
        for(size_t i=0; i<half; i++){
            float angle = static_cast<float>(positions[t]) * inv_freq[i];
            cos.at({t, i}) = std::cos(angle);
            cos.at({t, i + half}) = std::cos(angle);
            sin.at({t, i}) = std::sin(angle);
            sin.at({t, i + half}) = std::sin(angle);
        }
    }

}

void rope_sincos(size_t seq_len,
                 size_t dim,
                 Tensor<float>& cos,
                 Tensor<float>& sin,
                 size_t position_offset,
                 float theta) {
    ENSURE(seq_len > 0, std::invalid_argument, "rope_sincos: seq_len must be positive");
    ENSURE(dim > 0 && dim % 2 == 0, std::invalid_argument, "rope_sincos: dim must be positive and even");
    const std::vector<size_t> table_shape = {seq_len, dim};
    ENSURE(cos.shape() == table_shape && sin.shape() == table_shape,
           std::invalid_argument,
           "rope_sincos: cos and sin must have shape [seq_len, dim]");

    Tensor<float> inv_freq = rope_freqs(dim, theta);
    std::vector<size_t> positions(seq_len);
    for (size_t t = 0; t < seq_len; ++t) {
        positions[t] = position_offset + t;
    }
    rope_sincos(inv_freq, positions, cos, sin);
}

void rope(const Tensor<float>& x,
          const Tensor<float>& cos,
          const Tensor<float>& sin,
          Tensor<float>& out) {
    ENSURE(!x.empty(), std::invalid_argument, "rope: x cannot be empty");
    ENSURE(x.ndim() >= 2, std::invalid_argument, "rope: x must have shape [seq, ..., head_dim]");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "rope: out shape must match x");

    const size_t seq = x.shape()[0];
    const size_t heads = x.shape()[1];
    const size_t dim = x.shape().back();
    const size_t dim_half = dim/2;
    const std::vector<size_t> table_shape = {seq, dim};
    const size_t n_inner = x.size() / seq / dim;

    ENSURE(dim % 2 == 0, std::invalid_argument, "rope: head_dim (last dim) must be even");
    ENSURE(cos.shape() == table_shape, std::invalid_argument, "rope: cos must have shape [seq, head_dim]");
    ENSURE(sin.shape() == table_shape, std::invalid_argument, "rope: sin must have shape [seq, head_dim]");


    // x: [seq, ..., head_dim]  (row-major; last dim contiguous)
    // For each token t and each inner vector (head):
    //   x1, x2 = split last dim in half
    //   out1 = x1 * cos[t, :half] - x2 * sin[t, :half]
    //   out2 = x2 * cos[t, half:] + x1 * sin[t, half:]
    //
    // Equivalent:  out = x * cos + rotate_half(x) * sin
    //   rotate_half(x) = concat(-x2, x1)
    //
    // Tensor::row(i) is a 1-D view of the i-th last-dim vector (i = t * n_inner + h).
    // Broadcast cos/sin [seq, dim] across the inner dims at the same t.

    for(size_t t=0; t<seq;t++){
        for(size_t h=0; h<n_inner; h++){ // could be 1 dim (heads), or many dims, or no dim at all (if X is 2D)
            Tensor<float> row = x.row(t * n_inner + h);
            for(size_t d=0; d < dim_half; d++){
                float x1 = row[d];
                float x2 =  row[d + dim_half];
                float s = sin.at({t, d});
                float c = cos.at({t, d });
                Tensor<float> out_row = out.row(t * n_inner + h);
                out_row[d] = x1 * c - x2 * s;
                out_row[d + dim_half] = x2 * c + x1 * s;
            }
        }
    }
}

Tensor<float> rope(const Tensor<float>& x, const Tensor<float>& cos, const Tensor<float>& sin) {
    ENSURE(!x.empty(), std::invalid_argument, "rope: x cannot be empty");
    Tensor<float> out(x.shape());
    rope(x, cos, sin, out);
    return out;
}

void rope(const Tensor<float>& x, Tensor<float>& out, size_t position_offset, float theta) {
    ENSURE(!x.empty(), std::invalid_argument, "rope: x cannot be empty");
    ENSURE(x.ndim() >= 2, std::invalid_argument, "rope: x must have shape [seq, ..., head_dim]");
    ENSURE(out.shape() == x.shape(), std::invalid_argument, "rope: out shape must match x");
    const size_t dim = x.shape().back();
    ENSURE(dim % 2 == 0, std::invalid_argument, "rope: head_dim (last dim) must be even");

    const size_t seq = x.shape()[0];
    Tensor<float> cos({seq, dim});
    Tensor<float> sin({seq, dim});
    rope_sincos(seq, dim, cos, sin, position_offset, theta);
    rope(x, cos, sin, out);
}

Tensor<float> rope(const Tensor<float>& x, size_t position_offset, float theta) {
    ENSURE(!x.empty(), std::invalid_argument, "rope: x cannot be empty");
    Tensor<float> out(x.shape());
    rope(x, out, position_offset, theta);
    return out;
}

}  // namespace llmberry
