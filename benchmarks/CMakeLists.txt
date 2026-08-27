function(llmberry_add_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target_name} PRIVATE llmberry)
endfunction()

llmberry_add_benchmark(benchmark_matmul benchmark_matmul.cpp)
llmberry_add_benchmark(benchmark_attention benchmark_attention.cpp)
llmberry_add_benchmark(benchmark_decode benchmark_decode.cpp)
