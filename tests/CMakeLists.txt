include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

function(llmberry_add_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE llmberry GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

llmberry_add_test(test_tensor)
llmberry_add_test(test_matmul)
llmberry_add_test(test_rmsnorm)
llmberry_add_test(test_rope)
llmberry_add_test(test_attention)
