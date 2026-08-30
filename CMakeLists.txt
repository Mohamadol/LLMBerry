cmake_minimum_required(VERSION 3.16)
project(LLMBerry VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LLMBERRY_BUILD_TESTS "Build unit tests" ON)
option(LLMBERRY_BUILD_BENCHMARKS "Build benchmarks" ON)

add_library(llmberry STATIC
    src/tensor.cpp
    src/model.cpp
    src/transformer.cpp
    src/attention.cpp
    src/kv_cache.cpp
    src/scheduler.cpp
    src/sampling.cpp
    kernels/matmul.cpp
    kernels/gemv.cpp
    kernels/elemwise.cpp
    kernels/reduce.cpp
    kernels/rmsnorm.cpp
    kernels/rope.cpp
    kernels/softmax.cpp
    kernels/silu.cpp
    kernels/gelu.cpp
    kernels/quantized_matmul.cpp
)

target_include_directories(llmberry
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_executable(llmberry_cli src/main.cpp)
target_link_libraries(llmberry_cli PRIVATE llmberry)
set_target_properties(llmberry_cli PROPERTIES OUTPUT_NAME llmberry)

if(LLMBERRY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LLMBERRY_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
