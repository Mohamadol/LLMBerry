#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace llmberry {

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
};

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        default: return "unknown";
    }
}

template <typename T>
struct DTypeOf {
    static_assert(sizeof(T) == 0, "Unsupported tensor element type");
};

template <>
struct DTypeOf<float> {
    static constexpr DType value = DType::Float32;
};

template <>
struct DTypeOf<double> {
    static constexpr DType value = DType::Float64;
};

template <>
struct DTypeOf<int32_t> {
    static constexpr DType value = DType::Int32;
};

template <>
struct DTypeOf<int64_t> {
    static constexpr DType value = DType::Int64;
};

template <typename T>
constexpr DType dtype_of() {
    return DTypeOf<T>::value;
}

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32: return sizeof(int32_t);
        case DType::Int64: return sizeof(int64_t);
        default: throw std::invalid_argument("Unknown dtype");
    }
}

}  // namespace llmberry
