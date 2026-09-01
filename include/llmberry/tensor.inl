#pragma once

#include "llmberry/tensor.h"

#include <memory>
#include <numeric>
#include <stdexcept>

namespace llmberry {

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    validate_shape(shape_);
    compute_row_major_strides();

    storage_ = std::make_shared<std::vector<T>>(size());
    data_ = storage_->data();

    offset_ = 0;
    owns_data_ = true;
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape,
                  std::vector<size_t> strides,
                  T* data,
                  size_t offset,
                  std::shared_ptr<std::vector<T>> storage)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      data_(data),
      offset_(offset),
      storage_(std::move(storage)),
      owns_data_(false) {
    validate_shape(shape_);

    if (strides_.size() != shape_.size()) {
        throw std::invalid_argument("Strides rank must match shape rank");
    }

    if (size() > 0 && data_ == nullptr) {
        throw std::invalid_argument("Tensor data pointer cannot be null");
    }
}

template <typename T>
Tensor<T> Tensor<T>::zeros(std::vector<size_t> shape) {
    Tensor<T> tensor(std::move(shape));
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = T{0};
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::ones(std::vector<size_t> shape) {
    Tensor<T> tensor(std::move(shape));
    for(size_t i=0; i<tensor.size(); i++){
        tensor[i] = T{1};
    }
    return tensor;
}

template <typename T>
size_t Tensor<T>::size() const {
    if (shape_.empty()) {
        return 0;
    }
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
}

template <typename T>
size_t Tensor<T>::nbytes() const {
    return size() * sizeof(T);
}


template <typename T>
bool Tensor<T>::empty() const {
    return size()==0;
}

template <typename T>
T* Tensor<T>::data() {
    return data_ + offset_; 
}

template <typename T>
const T* Tensor<T>::data() const {
    return data_ + offset_; 
}

template <typename T>
T& Tensor<T>::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("index into the tensor is out of range.");
    }
    return data()[index];
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("index into the tensor is out of range.");
    }
    return data()[index];
}


template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
    validate_indices(indices);
    return data()[offset_from_indices(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
    validate_indices(indices);
    return data()[offset_from_indices(indices)];
}

template <typename T>
Tensor<T> Tensor<T>::view(std::vector<size_t> new_shape) const {
    validate_shape(new_shape);

    // confirm new shape matches the total size
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != size()) {
        throw std::invalid_argument("view shape must have the same number of elements");
    }

    std::vector<size_t> new_strides(new_shape.size(),1);
    for(auto i= static_cast<int>(new_shape.size())-2; i >= 0; i--){
        new_strides[i] = new_strides[i+1] * new_shape[i+1];
    }
    return Tensor<T>(new_shape, new_strides, data_, offset_, storage_);
}

template <typename T>
Tensor<T> Tensor<T>::row(size_t index) const {
    if (shape_.empty()) {
        throw std::invalid_argument("row() requires a non-empty tensor");
    }
    if (strides_.back() != 1) {
        throw std::invalid_argument("row() requires a contiguous last dimension");
    }

    const size_t dim = shape_.back();
    const size_t n_rows = size() / dim;
    if (index >= n_rows) {
        throw std::out_of_range("row index out of range");
    }

    return Tensor<T>({dim}, {1}, data_, offset_ + index * dim, storage_);
}

template <typename T>
Tensor<T> Tensor<T>::matrix(size_t index) const {
    if (shape_.empty()) {
        throw std::invalid_argument("matrix() requires a non-empty tensor");
    }

    if (ndim() < 2) {
        throw std::invalid_argument("matrix() requires at least 2D");
    }

    const size_t rows = shape_[ndim() - 2];
    const size_t cols = shape_[ndim() - 1];
    const size_t n_matrices = size() / (rows * cols);
    if (index >= n_matrices) {
        throw std::out_of_range("matrix index out of range");
    }
    const size_t plane_stride = (ndim() >= 3) ? strides_[ndim() - 3] : 0;
    return Tensor<T>(
        {rows, cols}, // view's shape
        {strides_[ndim() - 2], strides_[ndim() - 1]}, // view's strides
        data_,
        offset_ + index * plane_stride,
        storage_);
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    if (ndim() < 2) {
        throw std::invalid_argument("transpose() requires rank >= 2");
    }
    auto shape = shape_;
    auto strides = strides_;
    std::swap(shape[ndim() - 2], shape[ndim() - 1]);
    std::swap(strides[ndim() - 2], strides[ndim() - 1]);
    return Tensor<T>(std::move(shape), std::move(strides), data_, offset_, storage_);
}

template <typename T>
void Tensor<T>::validate_shape(const std::vector<size_t>& shape) const {
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
    }
}

template <typename T>
void Tensor<T>::compute_row_major_strides() {
    strides_.assign(shape_.size(), 1);
    if (shape_.empty()) {
        return;
    }
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

template <typename T>
size_t Tensor<T>::offset_from_indices(const std::vector<size_t>& indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}

template <typename T>
void Tensor<T>::validate_indices(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Index rank does not match tensor rank");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
    }
}

}  // namespace llmberry
