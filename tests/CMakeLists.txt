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
llmberry_add_test(test_mlp)
llmberry_add_test(test_transformer)

# JSON dump of kernel outputs for python/verify_ops.py (not a gtest).
add_executable(dump_ops dump_ops.cpp)
target_link_libraries(dump_ops PRIVATE llmberry)

if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME verify_ops
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/verify_ops.py
                --bin $<TARGET_FILE:dump_ops>
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME test_transformer_pytorch
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/test_transformer_pytorch.py
                --bin $<TARGET_FILE:dump_ops>
                --allow-skip
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
