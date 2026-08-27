# LLMBerry

A lightweight C++ runtime for LLM inference and decode optimization.

See [project.md](project.md) for the full roadmap.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

Or run a single test binary:

```bash
./build/tests/test_tensor
```

## Run CLI (scaffold)

```bash
./build/llmberry
```

## Checkpoint 01 — Tensor

Implement the methods marked `TODO` in `include/llmberry/tensor.inl`. Helper methods `validate_shape`, `compute_row_major_strides`, `size`, and `nbytes` are already provided.

When your implementation is correct, all tests in `tests/test_tensor.cpp` should pass.
