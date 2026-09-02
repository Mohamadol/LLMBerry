# LLMBerry — named commands for build, test, and benchmarks.
# Usage: make <target>

BUILD_DIR   := build
BUILD_TYPE  ?= Release
GENERATOR   ?= Unix Makefiles
CMAKE       := cmake
NPROC       := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)

TEST_TENSOR     := $(BUILD_DIR)/tests/test_tensor
TEST_MATMUL     := $(BUILD_DIR)/tests/test_matmul
TEST_RMSNORM    := $(BUILD_DIR)/tests/test_rmsnorm
TEST_ROPE       := $(BUILD_DIR)/tests/test_rope
TEST_ATTENTION  := $(BUILD_DIR)/tests/test_attention
TEST_MLP        := $(BUILD_DIR)/tests/test_mlp
DUMP_OPS        := $(BUILD_DIR)/tests/dump_ops
LLMBERRY        := $(BUILD_DIR)/llmberry

.PHONY: help setup build clean test verify test-tensor test-tensor-construction \
        test-tensor-indexing test-tensor-views test-matmul test-rmsnorm test-rope \
        test-attention test-mlp verify-ops benchmark run

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
	@echo "  make test-matmul            Run checkpoint 02 kernel tests"
	@echo "  make test-rmsnorm           Run checkpoint 04 RMSNorm tests"
	@echo "  make test-rope              Run checkpoint 05 RoPE tests"
	@echo "  make test-attention         Run checkpoint 06 attention tests"
	@echo "  make test-mlp               Run checkpoint 08 SwiGLU MLP tests"
	@echo "  make verify-ops             C++ dump_ops vs NumPy (and PyTorch if installed)"
	@echo "  make benchmark              Run benchmark binaries"
	@echo "  make run                    Run llmberry CLI"
	@echo "  make clean                  Remove build directory"

setup:
	@$(CMAKE) -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)

build:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)

clean:
	@rm -rf $(BUILD_DIR)

test:
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

verify:
	@$(CMAKE) -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

test-tensor:
	@$(MAKE) build
	@$(TEST_TENSOR)

test-tensor-construction:
	@$(MAKE) build
	@$(TEST_TENSOR) --gtest_filter='TensorConstruction.*'

test-tensor-indexing:
	@$(MAKE) build
	@$(TEST_TENSOR) --gtest_filter='TensorIndexing.*:TensorData.*:TensorLayout.*'

test-tensor-views:
	@$(MAKE) build
	@$(TEST_TENSOR) --gtest_filter='TensorView.*'

test-matmul:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) --target test_matmul -j$(NPROC)
	@$(TEST_MATMUL)

test-rmsnorm:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) --target test_rmsnorm -j$(NPROC)
	@$(TEST_RMSNORM)

test-rope:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) --target test_rope -j$(NPROC)
	@$(TEST_ROPE)

test-attention:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) --target test_attention -j$(NPROC)
	@$(TEST_ATTENTION)

test-mlp:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) --target test_mlp -j$(NPROC)
	@$(TEST_MLP)

verify-ops:
	@test -d $(BUILD_DIR) || $(MAKE) setup
	@$(CMAKE) --build $(BUILD_DIR) --target dump_ops -j$(NPROC)
	@python3 python/verify_ops.py --bin $(DUMP_OPS)

benchmark:
	@test -x $(BUILD_DIR)/benchmarks/benchmark_matmul || $(MAKE) build
	@$(BUILD_DIR)/benchmarks/benchmark_matmul
	@$(BUILD_DIR)/benchmarks/benchmark_attention
	@$(BUILD_DIR)/benchmarks/benchmark_decode

run:
	@test -x $(LLMBERRY) || $(MAKE) build
	@$(LLMBERRY)
