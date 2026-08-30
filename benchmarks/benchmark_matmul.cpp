#include "benchmark_common.h"
#include "llmberry/kernels.h"

#include <cstring>
#include <iostream>
#include <random>
#include <vector>

void fill_uniform(llmberry::Tensor<float>& t, float lo, float hi, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = dist(rng);
    }
}

struct MatmulShape {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
};

namespace llmberry {
namespace bench {

// drive the interface benchmarks use to store their results
struct MatmulResult final : BenchmarkResult {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;

    std::string name() const override { return "matmul"; }

    int threads() const override { return 1; }

    void write_header(std::ostream& os) const override {
        os << "Matrix: " << m << " x " << n << "  (K=" << k << ")\n";
    }

    nlohmann::json spec_json() const override {
        return {{"m", m}, {"n", n}, {"k", k}};
    }
};

}  // namespace bench
}  // namespace llmberry

int main(int argc, char** argv) {
    constexpr int kWarmup = 2;
    constexpr int kIters = 10;
    constexpr float kValuesLow = -40.0f;
    constexpr float kValuesHigh = 40.0f;

    bool large = false;
    bool json = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--large") == 0) {
            large = true;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " [--large] [--json]\n";
            return 1;
        }
    }

    std::vector<MatmulShape> shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {256, 768, 768},
    };
    if (large) {
        shapes.push_back({256, 4096, 4096});
    }

    std::vector<llmberry::bench::MatmulResult> results;
    results.reserve(shapes.size());

    for (const auto& s : shapes) {
        std::mt19937 rng(42);
        llmberry::Tensor<float> a({s.m, s.k});
        llmberry::Tensor<float> b({s.k, s.n});
        llmberry::Tensor<float> c({s.m, s.n});
        fill_uniform(a, kValuesLow, kValuesHigh, rng);
        fill_uniform(b, kValuesLow, kValuesHigh, rng);

        for (int i = 0; i < kWarmup; ++i) {
            llmberry::matmul(a, b, c);
        }

        llmberry::bench::Timer timer;
        timer.start();
        for (int i = 0; i < kIters; ++i) {
            llmberry::matmul(a, b, c);
        }
        const double elapsed_ms = timer.elapsed_ms() / kIters;
        const double flops = 2.0 * static_cast<double>(s.m) * s.n * s.k;
        const double gflops = llmberry::bench::gflops_from_elapsed_ms(flops, elapsed_ms);

        llmberry::bench::MatmulResult r;
        r.m = s.m;
        r.n = s.n;
        r.k = s.k;
        r.elapsed_ms = elapsed_ms;
        r.throughput_gflops = gflops;
        results.push_back(r);

        if (!json) {
            llmberry::bench::write_human(std::cout, r);
        }
    }

    if (json) {
        llmberry::bench::write_json_array(std::cout, results);
    }
    return 0;
}
