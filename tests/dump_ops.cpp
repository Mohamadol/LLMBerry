/// Dump C++ kernel outputs as JSON for python/verify_ops.py (C++ vs NumPy).
/// Prints one JSON object to stdout. Inputs are included so Python recomputes
/// the NumPy reference from the same arrays.

#include "llmberry/llmberry.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using llmberry::add;
using llmberry::attention;
using llmberry::dot;
using llmberry::gelu;
using llmberry::gemv;
using llmberry::matmul;
using llmberry::mlp;
using llmberry::mul;
using llmberry::reduce_mean;
using llmberry::reduce_sum;
using llmberry::rmsnorm;
using llmberry::rope;
using llmberry::rope_freqs;
using llmberry::rope_sincos;
using llmberry::silu;
using llmberry::softmax;
using llmberry::Tensor;
using llmberry::Transformer;
using llmberry::TransformerConfig;

void fill_arange(Tensor<float>& t, float start = 1.0f) {
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = start + static_cast<float>(i);
    }
}

void fill_uniform(Tensor<float>& t, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = dist(rng);
    }
}

void randomize_block(Transformer& block, std::mt19937& rng) {
    auto& w = block.weights();
    fill_uniform(w.attn_norm, rng, 0.5f, 1.5f);
    fill_uniform(w.w_q, rng, -0.2f, 0.2f);
    fill_uniform(w.w_k, rng, -0.2f, 0.2f);
    fill_uniform(w.w_v, rng, -0.2f, 0.2f);
    fill_uniform(w.w_o, rng, -0.2f, 0.2f);
    fill_uniform(w.ffn_norm, rng, 0.5f, 1.5f);
    fill_uniform(w.w_gate, rng, -0.2f, 0.2f);
    fill_uniform(w.w_up, rng, -0.2f, 0.2f);
    fill_uniform(w.w_down, rng, -0.2f, 0.2f);
}

void fill_eye(Tensor<float>& t) {
    const size_t rows = t.shape()[0];
    const size_t cols = t.shape()[1];
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t.at({i, j}) = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void write_tensor(std::ostream& os, const Tensor<float>& t) {
    os << "{\"shape\":[";
    for (size_t i = 0; i < t.ndim(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << t.shape()[i];
    }
    os << "],\"data\":[";
    os << std::setprecision(9);
    for (size_t i = 0; i < t.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << t[i];
    }
    os << "]}";
}

void write_named_tensor(std::ostream& os, const char* name, const Tensor<float>& t, bool comma) {
    if (comma) {
        os << ",";
    }
    os << "\"" << name << "\":";
    write_tensor(os, t);
}

int main() {
    std::vector<std::string> ops;
    auto push = [&](std::string s) { ops.push_back(std::move(s)); };

    {
        Tensor<float> a({2, 3});
        Tensor<float> b({3, 2});
        fill_arange(a);
        fill_arange(b);
        Tensor<float> c = Tensor<float>::zeros({2, 2});
        matmul(a, b, c);
        std::ostringstream os;
        os << "{\"name\":\"matmul\",\"inputs\":{";
        write_named_tensor(os, "a", a, false);
        write_named_tensor(os, "b", b, true);
        os << "},\"output\":";
        write_tensor(os, c);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> a({3, 4});
        Tensor<float> x({4});
        fill_arange(a);
        fill_arange(x);
        Tensor<float> out({3});
        gemv(a, x, out);
        std::ostringstream os;
        os << "{\"name\":\"gemv\",\"inputs\":{";
        write_named_tensor(os, "a", a, false);
        write_named_tensor(os, "x", x, true);
        os << "},\"output\":";
        write_tensor(os, out);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> a({2, 3});
        Tensor<float> b({2, 3});
        fill_arange(a, 1.0f);
        fill_arange(b, -2.0f);
        Tensor<float> sum = add(a, b);
        Tensor<float> prod = mul(a, b);
        {
            std::ostringstream os;
            os << "{\"name\":\"add\",\"inputs\":{";
            write_named_tensor(os, "a", a, false);
            write_named_tensor(os, "b", b, true);
            os << "},\"output\":";
            write_tensor(os, sum);
            os << "}";
            push(os.str());
        }
        {
            std::ostringstream os;
            os << "{\"name\":\"mul\",\"inputs\":{";
            write_named_tensor(os, "a", a, false);
            write_named_tensor(os, "b", b, true);
            os << "},\"output\":";
            write_tensor(os, prod);
            os << "}";
            push(os.str());
        }
    }

    {
        Tensor<float> a({4});
        Tensor<float> b({4});
        fill_arange(a);
        fill_arange(b, 0.5f);
        const float d = dot(a, b);
        std::ostringstream os;
        os << "{\"name\":\"dot\",\"inputs\":{";
        write_named_tensor(os, "a", a, false);
        write_named_tensor(os, "b", b, true);
        os << "},\"value\":" << std::setprecision(9) << d << "}";
        push(os.str());
    }

    {
        Tensor<float> x({2, 3});
        fill_arange(x);
        std::ostringstream sum_os;
        sum_os << "{\"name\":\"reduce_sum\",\"inputs\":{";
        write_named_tensor(sum_os, "x", x, false);
        sum_os << "},\"value\":" << std::setprecision(9) << reduce_sum(x) << "}";
        push(sum_os.str());
        std::ostringstream mean_os;
        mean_os << "{\"name\":\"reduce_mean\",\"inputs\":{";
        write_named_tensor(mean_os, "x", x, false);
        mean_os << "},\"value\":" << std::setprecision(9) << reduce_mean(x) << "}";
        push(mean_os.str());
    }

    {
        Tensor<float> x({2, 4});
        fill_arange(x, -3.0f);
        Tensor<float> y_silu = silu(x);
        Tensor<float> y_gelu = gelu(x);
        {
            std::ostringstream os;
            os << "{\"name\":\"silu\",\"inputs\":{";
            write_named_tensor(os, "x", x, false);
            os << "},\"output\":";
            write_tensor(os, y_silu);
            os << "}";
            push(os.str());
        }
        {
            std::ostringstream os;
            os << "{\"name\":\"gelu\",\"inputs\":{";
            write_named_tensor(os, "x", x, false);
            os << "},\"output\":";
            write_tensor(os, y_gelu);
            os << "}";
            push(os.str());
        }
    }

    {
        Tensor<float> x({2, 4});
        Tensor<float> w({4});
        fill_arange(x);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = 1.0f;
        }
        Tensor<float> y = rmsnorm(x, w);
        std::ostringstream os;
        os << "{\"name\":\"rmsnorm\",\"inputs\":{";
        write_named_tensor(os, "x", x, false);
        write_named_tensor(os, "weight", w, true);
        os << "},\"attrs\":{\"eps\":1e-6},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> inv = rope_freqs(4, 10000.0f);
        std::ostringstream os;
        os << "{\"name\":\"rope_freqs\",\"attrs\":{\"dim\":4,\"theta\":10000.0},\"output\":";
        write_tensor(os, inv);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> cos({2, 4});
        Tensor<float> sin({2, 4});
        rope_sincos(2, 4, cos, sin, /*position_offset=*/0, 10000.0f);
        std::ostringstream os;
        os << "{\"name\":\"rope_sincos\",\"attrs\":{\"seq_len\":2,\"dim\":4,\"position_offset\":0,\"theta\":10000.0},\"outputs\":{";
        write_named_tensor(os, "cos", cos, false);
        write_named_tensor(os, "sin", sin, true);
        os << "}}";
        push(os.str());
    }

    {
        Tensor<float> x({2, 4});
        fill_arange(x);
        Tensor<float> y = rope(x, /*position_offset=*/0, 10000.0f);
        std::ostringstream os;
        os << "{\"name\":\"rope\",\"inputs\":{";
        write_named_tensor(os, "x", x, false);
        os << "},\"attrs\":{\"position_offset\":0,\"theta\":10000.0},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> x({3});
        x[0] = 1.0f;
        x[1] = 2.0f;
        x[2] = 3.0f;
        Tensor<float> y = softmax(x);
        std::ostringstream os;
        os << "{\"name\":\"softmax\",\"inputs\":{";
        write_named_tensor(os, "x", x, false);
        os << "},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> qkv({2, 4});
        fill_arange(qkv);
        Tensor<float> y = attention(qkv, qkv, qkv, /*causal=*/true);
        std::ostringstream os;
        os << "{\"name\":\"attention\",\"inputs\":{";
        write_named_tensor(os, "q", qkv, false);
        write_named_tensor(os, "k", qkv, true);
        write_named_tensor(os, "v", qkv, true);
        os << "},\"attrs\":{\"causal\":true},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> q({4, 2, 4});
        Tensor<float> k({2, 2, 4});
        Tensor<float> v({2, 2, 4});
        fill_arange(q);
        fill_arange(k, 0.5f);
        fill_arange(v, -1.0f);
        Tensor<float> y = attention(q, k, v, /*causal=*/true);
        std::ostringstream os;
        os << "{\"name\":\"attention_gqa\",\"inputs\":{";
        write_named_tensor(os, "q", q, false);
        write_named_tensor(os, "k", k, true);
        write_named_tensor(os, "v", v, true);
        os << "},\"attrs\":{\"causal\":true},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> x({2, 4});
        Tensor<float> w_gate({4, 6});
        Tensor<float> w_up = Tensor<float>::ones({4, 6});
        Tensor<float> w_down({6, 4});
        fill_arange(x);
        fill_eye(w_gate);
        fill_eye(w_down);
        Tensor<float> y = mlp(x, w_gate, w_up, w_down);
        std::ostringstream os;
        os << "{\"name\":\"mlp\",\"inputs\":{";
        write_named_tensor(os, "x", x, false);
        write_named_tensor(os, "w_gate", w_gate, true);
        write_named_tensor(os, "w_up", w_up, true);
        write_named_tensor(os, "w_down", w_down, true);
        os << "},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    {
        Tensor<float> x({2, 3, 4});
        Tensor<float> w_gate({4, 6});
        Tensor<float> w_up({4, 6});
        Tensor<float> w_down({6, 4});
        fill_arange(x, -2.0f);
        fill_arange(w_gate, 0.1f);
        fill_arange(w_up, -0.2f);
        fill_arange(w_down, 0.05f);
        Tensor<float> y = mlp(x, w_gate, w_up, w_down);
        std::ostringstream os;
        os << "{\"name\":\"mlp_batched\",\"inputs\":{";
        write_named_tensor(os, "x", x, false);
        write_named_tensor(os, "w_gate", w_gate, true);
        write_named_tensor(os, "w_up", w_up, true);
        write_named_tensor(os, "w_down", w_down, true);
        os << "},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    }

    auto dump_transformer = [&](const char* name, Transformer& block, const Tensor<float>& x,
                                size_t position_offset = 0) {
        Tensor<float> y = block.forward(x, position_offset);
        const auto& w = block.weights();
        const auto& c = block.config();
        std::ostringstream os;
        os << "{\"name\":\"" << name << "\",\"inputs\":{";
        write_named_tensor(os, "x", x, false);
        write_named_tensor(os, "attn_norm", w.attn_norm, true);
        write_named_tensor(os, "w_q", w.w_q, true);
        write_named_tensor(os, "w_k", w.w_k, true);
        write_named_tensor(os, "w_v", w.w_v, true);
        write_named_tensor(os, "w_o", w.w_o, true);
        write_named_tensor(os, "ffn_norm", w.ffn_norm, true);
        write_named_tensor(os, "w_gate", w.w_gate, true);
        write_named_tensor(os, "w_up", w.w_up, true);
        write_named_tensor(os, "w_down", w.w_down, true);
        os << "},\"attrs\":{\"n_heads\":" << c.n_heads << ",\"n_kv_heads\":" << c.n_kv_heads
           << ",\"position_offset\":" << position_offset << ",\"theta\":" << c.rope_theta
           << ",\"eps\":" << c.rms_eps << "},\"output\":";
        write_tensor(os, y);
        os << "}";
        push(os.str());
    };

    {
        TransformerConfig cfg;
        cfg.hidden = 4;
        cfg.n_heads = 2;
        cfg.n_kv_heads = 2;
        cfg.intermediate = 6;
        Transformer block(cfg);
        fill_arange(block.weights().attn_norm, 0.5f);
        fill_arange(block.weights().w_q, 0.01f);
        fill_arange(block.weights().w_k, -0.02f);
        fill_arange(block.weights().w_v, 0.03f);
        fill_arange(block.weights().w_o, -0.04f);
        fill_arange(block.weights().ffn_norm, 0.6f);
        fill_arange(block.weights().w_gate, 0.05f);
        fill_arange(block.weights().w_up, -0.06f);
        fill_arange(block.weights().w_down, 0.07f);
        Tensor<float> x({3, 4});
        fill_arange(x, -1.0f);
        dump_transformer("transformer", block, x);
    }

    {
        TransformerConfig cfg;
        cfg.hidden = 8;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.intermediate = 5;
        Transformer block(cfg);
        fill_arange(block.weights().attn_norm, 0.4f);
        fill_arange(block.weights().w_q, 0.01f);
        fill_arange(block.weights().w_k, -0.02f);
        fill_arange(block.weights().w_v, 0.03f);
        fill_arange(block.weights().w_o, -0.04f);
        fill_arange(block.weights().ffn_norm, 0.5f);
        fill_arange(block.weights().w_gate, 0.05f);
        fill_arange(block.weights().w_up, -0.06f);
        fill_arange(block.weights().w_down, 0.07f);
        Tensor<float> x({2, 3, 8});
        fill_arange(x, -2.0f);
        dump_transformer("transformer_gqa", block, x);
    }

    {
        std::mt19937 rng(42);
        struct RandomCase {
            const char* name;
            size_t hidden;
            size_t n_heads;
            size_t n_kv_heads;
            size_t intermediate;
            size_t position_offset;
            std::vector<size_t> x_shape;
        };
        const RandomCase cases[] = {
            // Tiny (fast)
            {"transformer_random_0", 8, 4, 4, 16, 0, {4, 8}},
            {"transformer_random_1", 8, 4, 2, 12, 0, {3, 8}},
            {"transformer_random_2", 8, 4, 2, 12, 5, {8}},
            {"transformer_random_3", 8, 4, 2, 16, 0, {2, 3, 8}},
            {"transformer_random_4", 4, 2, 2, 8, 1, {5, 4}},
            // Mini Llama-3 GQA (32:8 → 4:1), head_dim=64, intermediate ≈ 3.5×H
            {"transformer_real_0", 256, 4, 1, 896, 0, {8, 256}},
            // Decode, RoPE offset as if KV cache already has 17 tokens
            {"transformer_real_1", 256, 4, 1, 896, 17, {256}},
            // Mini Llama-2 MHA, head_dim=64
            {"transformer_real_2", 256, 4, 4, 688, 0, {16, 256}},
            // Larger GQA (Llama 3 8B 32:8), head_dim=64
            {"transformer_real_3", 512, 8, 2, 1792, 0, {8, 512}},
            // Batched prefill
            {"transformer_real_4", 256, 4, 1, 896, 0, {2, 8, 256}},
            // Llama-2/3 7B–8B head_dim=128
            {"transformer_real_5", 256, 2, 2, 688, 0, {8, 256}},
        };
        for (const auto& c : cases) {
            TransformerConfig cfg;
            cfg.hidden = c.hidden;
            cfg.n_heads = c.n_heads;
            cfg.n_kv_heads = c.n_kv_heads;
            cfg.intermediate = c.intermediate;
            Transformer block(cfg);
            randomize_block(block, rng);
            Tensor<float> x(c.x_shape);
            fill_uniform(x, rng, -1.0f, 1.0f);
            dump_transformer(c.name, block, x, c.position_offset);
        }
    }

    std::cout << "{\"ops\":[";
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << ops[i];
    }
    std::cout << "]}\n";
    return 0;
}
