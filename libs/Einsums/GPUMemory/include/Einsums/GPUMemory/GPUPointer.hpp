//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>

#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums {

namespace gpu {

/**
 * @struct GPUPointer
 *
 * @brief A pointer to memory on the GPU.
 *
 * This is used to wrap up certain operations. Unfortunately, it can not be used as a normal fancy pointer since
 * the @c addressof operation does not make sense for its references, which actually handle the data transfers.
 */
template <typename T>
struct GPUPointer final {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The type of data used by the GPU.
     *
     * For real data types, this is the same as on the host. For complex data types, the appropriate
     * HIP data type needs to be used.
     */
    using dev_datatype =
        std::conditional_t<std::is_void_v<T>,
                           std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, std::complex<float>>, hipFloatComplex,
                                              std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, std::complex<double>>,
                                                                 hipDoubleComplex, std::remove_cvref_t<T>>>,
                           T>;

    using difference_type   = ptrdiff_t;
    using value_type        = T;
    using element_type      = dev_datatype;
    using pointer           = GPUPointer<T>;
    using reference         = T &;
    using iterator_category = std::contiguous_iterator_tag;

    template <class U>
    using rebind = GPUPointer<U>;

    constexpr GPUPointer() = default;

    constexpr GPUPointer(GPUPointer<T> const &) = default;

    constexpr GPUPointer(GPUPointer<T> &&) = default;

    template <typename U>
        requires(!std::is_const_v<U>)
    constexpr GPUPointer(GPUPointer<U> const &other) : gpu_ptr_{other.gpu_ptr_} {}

    constexpr GPUPointer(dev_datatype *other) : gpu_ptr_{other} {}

    constexpr GPUPointer(std::nullptr_t) : gpu_ptr_{nullptr} {}

    constexpr ~GPUPointer() = default;

    constexpr GPUPointer<T> &operator=(std::nullptr_t) { gpu_ptr_ = nullptr; }

    constexpr GPUPointer<T> &operator=(GPUPointer<T> const &other) = default;

    constexpr GPUPointer<T> &operator=(GPUPointer<T> &&other) = default;

    template <typename U>
    constexpr GPUPointer<T> &operator=(GPUPointer<U> const &other) {
        gpu_ptr_ = other.gpu_ptr_;

        return *this;
    }

    template <typename U>
    constexpr GPUPointer<T> &operator=(GPUPointer<U> &&other) {
        gpu_ptr_ = other.gpu_ptr_;

        return *this;
    }

    template <typename U>
    constexpr GPUPointer<T> &operator=(GPUPointer<U const> const &other) = delete;

    template <typename U>
    constexpr GPUPointer<T> &operator=(GPUPointer<U const> &&other) = delete;

    constexpr bool operator==(GPUPointer<T> const &other) const { return gpu_ptr_ == other.gpu_ptr_; }

    template <typename U>
    constexpr bool operator==(GPUPointer<U> const &other) const {
        return gpu_ptr_ == other.gpu_ptr_;
    }

    constexpr bool operator==(std::nullptr_t) const { return gpu_ptr_ == nullptr; }

    template <typename U>
    constexpr bool operator==(U const *other) const {
        return gpu_ptr_ == other;
    }

    reference operator*() { return *this; }

    T *operator->() { return gpu_ptr_; }

    constexpr pointer &operator++() {
        gpu_ptr_++;
        return gpu_ptr_;
    }

    constexpr pointer &operator++(int) {
        gpu_ptr_++;
        return gpu_ptr_;
    }

    constexpr pointer &operator--() {
        gpu_ptr_--;
        return gpu_ptr_;
    }

    constexpr pointer &operator--(int) {
        gpu_ptr_--;
        return gpu_ptr_;
    }

    constexpr pointer &operator+=(difference_type &&n) {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    constexpr pointer &operator-=(difference_type &&n) {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    constexpr pointer operator+(difference_type &&n) const { return gpu_ptr_ + n; }

    constexpr pointer operator-(difference_type &&n) const { return gpu_ptr_ - n; }

    constexpr pointer operator-(GPUPointer<T> const &other) const { return gpu_ptr_ - other.gpu_ptr_; }

    constexpr reference operator[](difference_type &&n) { return gpu_ptr_ + n; }

    constexpr bool operator<(GPUPointer<T> const &other) const { return gpu_ptr_ < other.gpu_ptr_; }

    constexpr bool operator>(GPUPointer<T> const &other) const { return gpu_ptr_ > other.gpu_ptr_; }

    constexpr bool operator<=(GPUPointer<T> const &other) const { return gpu_ptr_ <= other.gpu_ptr_; }

    constexpr bool operator>=(GPUPointer<T> const &other) const { return gpu_ptr_ >= other.gpu_ptr_; }

    constexpr operator T *() { return static_cast<T *>(gpu_ptr_); }

    constexpr operator T const *() const { return static_cast<T const *>(gpu_ptr_); }

    template <typename U>
    constexpr operator U *() {
        return static_cast<U *>(gpu_ptr_);
    }

    static pointer pointer_to(element_type &other) { return pointer(&other); }

  private:
    dev_datatype *gpu_ptr_{nullptr};

    template <typename U>
    friend void std::swap(GPUPointer<U> &, GPUPointer<U> &);
};

#ifndef DOXYGEN
template <typename T>
struct GPUPointer<T const> final {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The type of data used by the GPU.
     *
     * For real data types, this is the same as on the host. For complex data types, the appropriate
     * HIP data type needs to be used.
     */
    using dev_datatype =
        std::conditional_t<std::is_void_v<T>,
                           std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, std::complex<float>>, hipFloatComplex,
                                              std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, std::complex<double>>,
                                                                 hipDoubleComplex, std::remove_cvref_t<T>>>,
                           T>;

    using difference_type   = ptrdiff_t;
    using element_type      = dev_datatype;
    using value_type        = T;
    using pointer           = GPUPointer<T const>;
    using reference         = T const &;
    using iterator_category = std::contiguous_iterator_tag;

    template <class U>
    using rebind = GPUPointer<U const>;

    constexpr GPUPointer() = default;

    constexpr GPUPointer(GPUPointer<T const> const &) = default;

    constexpr GPUPointer(GPUPointer<T const> &&) = default;

    template <typename U>
    constexpr GPUPointer(GPUPointer<U> const &other) : gpu_ptr_{other.gpu_ptr_} {}

    constexpr GPUPointer(dev_datatype const *other) : gpu_ptr_{other} {}

    constexpr GPUPointer(std::nullptr_t) : gpu_ptr_{nullptr} {}

    constexpr GPUPointer<T const> &operator=(std::nullptr_t) { gpu_ptr_ = nullptr; }

    constexpr GPUPointer<T const> &operator=(GPUPointer<T const> const &other) = default;

    template <typename U>
    constexpr GPUPointer<T const> &operator=(GPUPointer<U const> const &other) {
        gpu_ptr_ = other.gpu_ptr_;

        return *this;
    }

    template <typename U>
    constexpr GPUPointer<T const> &operator=(U const *other) {
        gpu_ptr_ = other;

        return *this;
    }

    constexpr bool operator==(GPUPointer<T const> const &other) const { return gpu_ptr_ == other.gpu_ptr_; }

    template <typename U>
    constexpr bool operator==(GPUPointer<U> const &other) const {
        return gpu_ptr_ == other.gpu_ptr_;
    }

    constexpr bool operator==(std::nullptr_t) const { return gpu_ptr_ == nullptr; }

    reference operator*() const { return *gpu_ptr_; }

    // Only provided for comaptibility. Do not use unless you can get objects onto the gpu.
    T const *operator->() const { return gpu_ptr_; }

    constexpr pointer operator++() {
        gpu_ptr_++;
        return gpu_ptr_;
    }

    constexpr pointer operator++(int) {
        gpu_ptr_++;
        return gpu_ptr_;
    }

    constexpr pointer operator--() {
        gpu_ptr_--;
        return gpu_ptr_;
    }

    constexpr pointer operator--(int) {
        gpu_ptr_--;
        return gpu_ptr_;
    }

    constexpr pointer operator+=(difference_type &&n) {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    constexpr pointer operator-=(difference_type &&n) {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    constexpr pointer operator+(difference_type &&n) const { return gpu_ptr_ + n; }

    constexpr pointer operator-(difference_type &&n) const { return gpu_ptr_ - n; }

    constexpr pointer operator-(GPUPointer<T> const &other) const { return gpu_ptr_ - other.gpu_ptr_; }

    constexpr reference operator[](difference_type &&n) { return gpu_ptr_ + n; }

    constexpr bool operator<(GPUPointer<T> const &other) const { return gpu_ptr_ < other.gpu_ptr_; }

    constexpr bool operator>(GPUPointer<T> const &other) const { return gpu_ptr_ > other.gpu_ptr_; }

    constexpr bool operator<=(GPUPointer<T> const &other) const { return gpu_ptr_ <= other.gpu_ptr_; }

    constexpr bool operator>=(GPUPointer<T> const &other) const { return gpu_ptr_ >= other.gpu_ptr_; }

    constexpr operator T const *() const { return gpu_ptr_; }

    template <typename U>
    constexpr operator U const *() const {
        return gpu_ptr_;
    }

  private:
    dev_datatype const *gpu_ptr_{nullptr};

    template <typename U>
    friend void std::swap(GPUPointer<U const> &, GPUPointer<U const> &);
};
#endif

} // namespace gpu

} // namespace einsums

template <typename T>
constexpr bool operator==(std::nullptr_t const &first, einsums::gpu::GPUPointer<T> const &second) {
    return second == first;
}

template <typename T>
constexpr bool operator+(ptrdiff_t &&offset, einsums::gpu::GPUPointer<T> const &base) {
    return base + offset;
}

namespace std {
template <typename T>
void swap(einsums::gpu::GPUPointer<T> &a, einsums::gpu::GPUPointer<T> &b) {
    auto *temp = a.gpu_ptr_;
    a.gpu_ptr_ = b.gpu_ptr_;
    b.gpu_ptr_ = a.gpu_ptr_;
}

template <typename T>
einsums::gpu::GPUPointer<T> memcpy(einsums::gpu::GPUPointer<T> &dest, T const *src, size_t count) {
    einsums::hip_catch(hipMemcpy((void *)dest, src, count, hipMemcpyHostToDevice));

    return dest;
}

template <typename T>
T *memcpy(T *dest, einsums::gpu::GPUPointer<T const> const &src, size_t count) {
    einsums::hip_catch(hipMemcpy(static_cast<void *>(dest), (void *const)src, count, hipMemcpyDeviceToHost));

    return dest;
}

template <typename T>
einsums::gpu::GPUPointer<T> memcpy(einsums::gpu::GPUPointer<T> &dest, einsums::gpu::GPUPointer<T const> const &src, size_t count) {
    einsums::hip_catch(hipMemcpy((void *)dest, (void const *)src, count, hipMemcpyDeviceToDevice));

    return dest;
}
} // namespace std