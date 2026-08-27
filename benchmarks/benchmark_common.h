#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace llmberry {
namespace bench {

struct BenchmarkResult {
    std::string name;
    double elapsed_ms = 0.0;
    double throughput_gflops = 0.0;
};

class Timer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }

    double elapsed_ms() const {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start_;
        return elapsed.count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

inline double gflops_from_elapsed_ms(double flops, double elapsed_ms) {
    if (elapsed_ms <= 0.0) {
        return 0.0;
    }
    return flops / (elapsed_ms * 1e6);
}

}  // namespace bench
}  // namespace llmberry
