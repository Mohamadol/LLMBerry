# LLMBerry — named commands for build, test, and benchmarks.
# Usage: make <target>

BUILD_DIR   := build
BUILD_TYPE  ?= Release
CMAKE       := cmake
NPROC       := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)

TEST_TENSOR := $(BUILD_DIR)/tests/test_tensor
LLMBERRY    := $(BUILD_DIR)/llmberry

.PHONY: help setup build clean test verify test-tensor test-tensor-construction \
        test-tensor-indexing test-tensor-views benchmark run

help:
	@echo "LLMBerry commands:"
	@echo "  make setup                  Configure and build (Release)"
	@echo "  make build                  Build only"
	@echo "  make test                   Run all tests (ctest)"
	@echo "  make verify                 Build + run all tests"
	@echo "  make test-tensor            Run tensor unit tests"
	@echo "  make test-tensor-construction"
	@echo "  make test-tensor-indexing"
	@echo "  make test-tensor-views"
	@echo "  make benchmark              Run benchmark binaries"
	@echo "  make run                    Run llmberry CLI"
	@echo "  make clean                  Remove build directory"

setup:
	@$(CMAKE) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)

build:
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)

clean:
	@rm -rf $(BUILD_DIR)

test:
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

verify:
	@$(CMAKE) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) 2>/dev/null || true
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

test-tensor:
	@test -x $(TEST_TENSOR) || $(MAKE) build
	@$(TEST_TENSOR)

test-tensor-construction:
	@test -x $(TEST_TENSOR) || $(MAKE) build
	@$(TEST_TENSOR) --gtest_filter='TensorConstruction.*'

test-tensor-indexing:
	@test -x $(TEST_TENSOR) || $(MAKE) build
	@$(TEST_TENSOR) --gtest_filter='TensorIndexing.*:TensorData.*:TensorLayout.*'

test-tensor-views:
	@test -x $(TEST_TENSOR) || $(MAKE) build
	@$(TEST_TENSOR) --gtest_filter='TensorView.*'

benchmark:
	@test -x $(BUILD_DIR)/benchmarks/benchmark_matmul || $(MAKE) build
	@$(BUILD_DIR)/benchmarks/benchmark_matmul
	@$(BUILD_DIR)/benchmarks/benchmark_attention
	@$(BUILD_DIR)/benchmarks/benchmark_decode

run:
	@test -x $(LLMBERRY) || $(MAKE) build
	@$(LLMBERRY)
