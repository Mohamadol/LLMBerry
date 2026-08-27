#pragma once

#include <numeric>
#include <stdexcept>

namespace llmberry {

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    validate_shape(shape_);
    compute_row_major_strides();

    // TODO: allocate storage_ and set data_/offset_/owns_data_
    throw std::runtime_error("Tensor constructor not implemented");
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape,
                  std::vector<size_t> strides,
                  T* data,
                  size_t offset)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      data_(data),
      offset_(offset),
      owns_data_(false) {
    validate_shape(shape_);
    // TODO: validate strides_, data_, and offset_
    throw std::runtime_error("Tensor view constructor not implemented");
}

template <typename T>
Tensor<T> Tensor<T>::zeros(std::vector<size_t> shape) {
    // TODO: construct tensor and fill with 0
    (void)shape;
    throw std::runtime_error("Tensor::zeros not implemented");
}

template <typename T>
Tensor<T> Tensor<T>::ones(std::vector<size_t> shape) {
    // TODO: construct tensor and fill with 1
    (void)shape;
    throw std::runtime_error("Tensor::ones not implemented");
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
T* Tensor<T>::data() {
    // TODO: return pointer to first logical element
    throw std::runtime_error("Tensor::data not implemented");
}

template <typename T>
const T* Tensor<T>::data() const {
    // TODO: return pointer to first logical element
    throw std::runtime_error("Tensor::data not implemented");
}

template <typename T>
T& Tensor<T>::operator[](size_t index) {
    // TODO: bounds check and return element at flat index
    (void)index;
    throw std::runtime_error("Tensor::operator[] not implemented");
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const {
    // TODO: bounds check and return element at flat index
    (void)index;
    throw std::runtime_error("Tensor::operator[] not implemented");
}

template <typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
    // TODO: validate indices and return element
    (void)indices;
    throw std::runtime_error("Tensor::at not implemented");
}

template <typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
    // TODO: validate indices and return element
    (void)indices;
    throw std::runtime_error("Tensor::at not implemented");
}

template <typename T>
Tensor<T> Tensor<T>::view(std::vector<size_t> new_shape) const {
    // TODO: return non-owning view sharing this tensor's storage
    (void)new_shape;
    throw std::runtime_error("Tensor::view not implemented");
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
    // TODO: compute byte/element offset from multi-dimensional indices
    (void)indices;
    throw std::runtime_error("Tensor::offset_from_indices not implemented");
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
