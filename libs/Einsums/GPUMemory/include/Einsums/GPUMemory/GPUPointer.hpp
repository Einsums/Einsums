//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUMemory/InitModule.hpp>
#include <Einsums/TypeSupport/TypeSwitch.hpp>

#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

namespace einsums {

namespace gpu {

/**
 * @struct GPUPointer
 *
 * @brief A pointer to memory on the GPU.
 *
 * This is used to wrap up certain operations. Unfortunately, it can not be used as a normal fancy pointer since
 * the @c addressof operation does not make sense for its references, which actually handle the data transfers.
 * However, for the majority of use cases, it can be used as a fancy pointer.
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
    using dev_datatype = SwitchT<std::remove_cv_t<T>, Case<hipFloatComplex, std::complex<float>>,
                                 Case<hipDoubleComplex, std::complex<double>>, Default<T>>;

    using difference_type   = ptrdiff_t;
    using value_type        = SwitchT<std::remove_cv_t<T>, Case<uint8_t, void>, Default<T>>;
    using element_type      = dev_datatype;
    using pointer           = GPUPointer<T>;
    using reference         = SwitchT<T, Case<uint8_t &, void>, Case<uint8_t const &, void const>, Default<std::add_lvalue_reference<T>>>;
    using iterator_category = std::contiguous_iterator_tag;

    template <class U>
    using rebind = GPUPointer<U>;

    constexpr GPUPointer() = default;

    constexpr GPUPointer(GPUPointer<T> const &) = default;

    constexpr GPUPointer(GPUPointer<T> &&) = default;

    constexpr GPUPointer(dev_datatype *other) : gpu_ptr_{other} {}

    template<typename U>
    constexpr GPUPointer(U *other) : gpu_ptr_{(dev_datatype *) other} {}

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

    template<std::integral Int>
    constexpr pointer &operator+=(Int n) {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    template<std::integral Int>
    constexpr pointer &operator-=(Int n) {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    template<std::integral Int>
    constexpr pointer operator+(Int n) const { return ((value_type *) gpu_ptr_) + n; }

    template<std::integral Int>
    constexpr pointer operator-(Int n) const { return ((value_type *) gpu_ptr_) - n; }

    constexpr pointer operator-(GPUPointer<T> const &other) const { return ((value_type *) gpu_ptr_) - ((value_type *) other.gpu_ptr_); }

    template<std::integral Int>
    constexpr reference operator[](Int n) { return *(((value_type *) gpu_ptr_) + n); }

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

    constexpr operator bool() const noexcept { return gpu_ptr_ == nullptr; }

    static pointer pointer_to(value_type &other) {
        return pointer(&other);
    }

  private:
    dev_datatype *gpu_ptr_{nullptr};

    template <typename U>
    friend void std::swap(GPUPointer<U> &, GPUPointer<U> &);
};

} // namespace gpu

} // namespace einsums

template <typename T>
constexpr bool operator==(std::nullptr_t const &first, einsums::gpu::GPUPointer<T> const &second) {
    return second == first;
}

template <typename T, std::integral Int>
constexpr bool operator+(Int &&offset, einsums::gpu::GPUPointer<T> const &base) {
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
einsums::gpu::GPUPointer<T> memcpy(einsums::gpu::GPUPointer<T> &dest, T const *src, size_t bytes) {
    einsums::hip_catch(hipMemcpy((void *)dest, src, bytes, hipMemcpyHostToDevice));

    return dest;
}

template <typename T>
T *memcpy(T *dest, einsums::gpu::GPUPointer<T const> const &src, size_t bytes) {
    einsums::hip_catch(hipMemcpy(static_cast<void *>(dest), (void *const)src, bytes, hipMemcpyDeviceToHost));

    return dest;
}

template <typename T>
einsums::gpu::GPUPointer<T> memcpy(einsums::gpu::GPUPointer<T> &dest, einsums::gpu::GPUPointer<T const> const &src, size_t bytes) {
    einsums::hip_catch(hipMemcpy((void *)dest, (void const *)src, bytes, hipMemcpyDeviceToDevice));

    return dest;
}
} // namespace std