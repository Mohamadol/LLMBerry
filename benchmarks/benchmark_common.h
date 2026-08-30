#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <ostream>
#include <string>
#include <vector>

namespace llmberry {
namespace bench {

/// Shared benchmark record. Each op subclasses and overrides name, threads,
/// write_header, and spec_json. Timing fields are filled after the run.
struct BenchmarkResult {
    double elapsed_ms = 0.0;
    double throughput_gflops = 0.0;

    virtual ~BenchmarkResult() = default;

    virtual std::string name() const = 0;
    virtual int threads() const = 0;
    virtual void write_header(std::ostream& os) const = 0;
    virtual nlohmann::json spec_json() const = 0;
};

inline void to_json(nlohmann::json& j, const BenchmarkResult& r) {
    j = r.spec_json();
    j["name"] = r.name();
    j["threads"] = r.threads();
    j["elapsed_ms"] = r.elapsed_ms;
    j["throughput_gflops"] = r.throughput_gflops;
}

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

template <typename Result>
void write_json_array(std::ostream& os, const std::vector<Result>& results) {
    os << nlohmann::json(results).dump(2) << "\n";
}

inline void write_human(std::ostream& os, const BenchmarkResult& r) {
    r.write_header(os);
    os << "Threads: " << r.threads() << "\n"
       << "Latency:        " << r.elapsed_ms << " ms\n"
       << "Throughput:     " << r.throughput_gflops << " GFLOP/s\n\n";
}

}  // namespace bench
}  // namespace llmberry
