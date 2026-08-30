#pragma once

#include "llmberry/dtype.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llmberry {

/// Lightweight row-major tensor with optional non-owning views.
///
/// Example:
///   Tensor<float> x({2, 4, 8});
template <typename T>
class Tensor {
public:
    Tensor() = default;

    /// Allocate a new tensor with the given shape.
    /// Strides should be computed for row-major layout.
    explicit Tensor(std::vector<size_t> shape);

    /// Create a tensor that views an existing buffer (does not own memory).
    /// Pass storage when sharing an owning tensor's buffer; omit for external memory.
    Tensor(std::vector<size_t> shape,
           std::vector<size_t> strides,
           T* data,
           size_t offset = 0,
           std::shared_ptr<std::vector<T>> storage = nullptr);

    static Tensor<T> zeros(std::vector<size_t> shape);
    static Tensor<T> ones(std::vector<size_t> shape);

    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }

    size_t ndim() const { return shape_.size(); }
    size_t size() const;   // total number of elements
    size_t nbytes() const; // size() * sizeof(T)
    bool empty() const; // size() == 0 ?

    DType dtype() const { return dtype_of<T>(); }

    T* data();
    const T* data() const;

    bool owns_data() const { return owns_data_; }
    bool is_view() const { return !owns_data_; }

    /// Flat element access using the tensor's logical indexing order.
    T& operator[](size_t index);
    const T& operator[](size_t index) const;

    /// Multi-dimensional element access.
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;

    /// Return a non-owning view with the same underlying storage.
  /// new_shape must have the same number of elements as this tensor.
    Tensor<T> view(std::vector<size_t> new_shape) const;

private:
    void validate_shape(const std::vector<size_t>& shape) const;
    void compute_row_major_strides();
    size_t offset_from_indices(const std::vector<size_t>& indices) const;
    void validate_indices(const std::vector<size_t>& indices) const;

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    std::shared_ptr<std::vector<T>> storage_;  // owned buffer (if any)
    T* data_ = nullptr;                        // start of logical data
    size_t offset_ = 0;                        // offset into storage_

    bool owns_data_ = false;
};

}  // namespace llmberry

#include "llmberry/tensor.inl"
