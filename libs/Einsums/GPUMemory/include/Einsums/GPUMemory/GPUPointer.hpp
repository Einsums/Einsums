//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
// #include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUMemory/InitModule.hpp>

#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums {

namespace gpu {

namespace detail {

template <typename T>
struct reference_or_void {
    using type = T &;
};

template <>
struct reference_or_void<void> {
    using type = uint8_t &;
};

template <>
struct reference_or_void<void const> {
    using type = uint8_t const &;
};

template <>
struct reference_or_void<void volatile> {
    using type = uint8_t volatile &;
};

template <>
struct reference_or_void<void const volatile> {
    using type = uint8_t const volatile &;
};

template <typename T>
using reference_or_void_t = typename reference_or_void<T>::type;

} // namespace detail

/**
 * @struct GPUPointer
 *
 * @brief A pointer to memory on the GPU.
 *
 * This is used to wrap up certain operations. Unfortunately, it can not be used as a normal fancy pointer since
 * the @c addressof operation does not make sense for its references, which actually handle the data transfers.
 * However, for the majority of use cases, it can be used as a fancy pointer.
 *
 * @tparam The host-side type held by the pointer.
 *
 * @versionadded{1.1.0}
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
     *
     * @versionadded{1.1.0}
     */
    using dev_datatype =
        std::conditional_t<std::is_void_v<T>,
                           std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, std::complex<float>>, hipFloatComplex,
                                              std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, std::complex<double>>,
                                                                 hipDoubleComplex, std::remove_cvref_t<T>>>,
                           T>;

    /**
     * @typedef difference_type
     *
     * @brief Type used for distances between pointers.
     *
     * @versionadded{1.1.0}
     */
    using difference_type = ptrdiff_t;

    /**
     * @typedef value_type
     *
     * @brief Data type referenced by this pointer.
     *
     * @versionadded{1.1.0}
     */
    using value_type = T;

    /**
     * @typedef element_type
     *
     * @brief Data type stored on the GPU.
     *
     * @versionadded{1.1.0}
     */
    using element_type = std::conditional_t<std::is_void_v<dev_datatype>, uint8_t, dev_datatype>;

    /**
     * @typedef pointer
     *
     * @brief Pointer type returned by various methods.
     *
     * @versionadded{1.1.0}
     */
    using pointer = GPUPointer<T>;

    /**
     * @typedef reference
     *
     * @brief Type of reference returned by various methods.
     *
     * @versionadded{1.1.0}
     */
    using reference = detail::reference_or_void_t<T>;

    /**
     * @typedef iterator_category
     *
     * @brief Indicates what kind of iterator this is.
     *
     * @versionadded{1.1.0}
     */
    using iterator_category = std::contiguous_iterator_tag;

    /**
     * @typedef rebind
     *
     * @brief Type of pointer of the same class that stores a different data type.
     *
     * @versionadded{1.1.0}
     */
    template <class U>
    using rebind = GPUPointer<U>;

    constexpr GPUPointer() noexcept = default;

    /**
     * @brief Copy constructor.
     *
     * @param[in] copy The pointer to copy.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer(GPUPointer<T> const &copy) noexcept = default;

    /**
     * @brief Move constructor.
     *
     * @param[inout] move The pointer to move. On exit, the input may be invalidated.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer(GPUPointer<T> &&move) noexcept = default;

    /**
     * @brief Copy cast constructor.
     *
     * This makes a new pointer pointing to the same data as another, but with a different data type.
     *
     * @tparam U The type of the other pointer.
     * @param[in] other The pointer to cast.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
        requires(std::is_const_v<T> || !std::is_const_v<U>)
    constexpr GPUPointer(GPUPointer<U> const &other) noexcept : gpu_ptr_{other.gpu_ptr_} {}

    /**
     * @brief Wrap a device pointer in a smart pointer.
     *
     * @param[in] other The pointer to wrap.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer(dev_datatype *other) noexcept : gpu_ptr_{other} {}

    /**
     * @brief Wrap a device pointer in a smart pointer.
     *
     * @param[in] other The pointer to wrap.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer(dev_datatype const *other) noexcept
        requires(std::is_const_v<T>)
        : gpu_ptr_{other} {}

    /**
     * @brief Wrap a null pointer in a smart pointer.
     *
     * @param[in] null A null pointer.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer(std::nullptr_t null) noexcept : gpu_ptr_{nullptr} {}

    constexpr ~GPUPointer() noexcept = default;

    /**
     * @brief Set the stored pointer to the null pointer.
     *
     * @param[in] null The null pointer.
     *
     * @return A reference to this.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer<T> &operator=(std::nullptr_t null) noexcept {
        gpu_ptr_ = nullptr;
        return *this;
    }

    /**
     * @brief Set the stored pointer to the pointer in another smart pointer.
     *
     * @param[in] other The other pointer to copy.
     *
     * @return A reference to this.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer<T> &operator=(GPUPointer<T> const &other) noexcept = default;

    /**
     * @brief Set the stored pointer to the pointer in another smart pointer and possibly invalidate the other pointer.
     *
     * @param[in] other The other pointer to move.
     *
     * @return A reference to this.
     *
     * @versionadded{1.1.0}
     */
    constexpr GPUPointer<T> &operator=(GPUPointer<T> &&other) noexcept = default;

    /**
     * @brief Set the stored pointer to the pointer in another smart pointer, casting to a new type.
     *
     * @tparam U The type stored by the other pointer.
     * @param[in] other The other pointer to copy.
     *
     * @return A reference to this.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
        requires(std::is_const_v<T> || !std::is_const_v<U>)
    constexpr GPUPointer<T> &operator=(GPUPointer<U> const &other) noexcept {
        gpu_ptr_ = other.gpu_ptr_;

        return *this;
    }

    /**
     * @brief Set the stored pointer to the pointer in another smart pointer, casting to a new type, and possibly invalidating the other
     * pointer.
     *
     * @tparam U The type stored by the other pointer.
     * @param[in] other The other pointer to move.
     *
     * @return A reference to this.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
        requires(std::is_const_v<T> || !std::is_const_v<U>)
    constexpr GPUPointer<T> &operator=(GPUPointer<U> &&other) noexcept {
        gpu_ptr_ = other.gpu_ptr_;

        return *this;
    }

    /**
     * @brief Check to see if the pointers in two objects are the same.
     *
     * @param[in] other The other pointer to check.
     *
     * @return True if the smart pointers reference the same data.
     *
     * @versionadded{1.1.0}
     */
    constexpr bool operator==(GPUPointer<T> const &other) const noexcept { return gpu_ptr_ == other.gpu_ptr_; }

    /**
     * @brief Check to see if the pointers in two objects are the same.
     *
     * @param[in] other The other pointer to check.
     *
     * @return True if the smart pointers reference the same data.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
    constexpr bool operator==(GPUPointer<U> const &other) const noexcept {
        return gpu_ptr_ == other.gpu_ptr_;
    }

    /**
     * @brief Check to see if this contains the null pointer.
     *
     * @param[in] other The null pointer.
     *
     * @return True if this contains the null pointer.
     *
     * @versionadded{1.1.0}
     */
    constexpr bool operator==(std::nullptr_t other) const noexcept { return gpu_ptr_ == nullptr; }

    /**
     * @brief Check to see if this smart pointer contains a certain bare pointer.
     *
     * @param[in] other The bare pointer to check.
     *
     * @return True if the smart pointer references the other pointer.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
    constexpr bool operator==(U const *other) const noexcept {
        return gpu_ptr_ == other;
    }

    /**
     * @brief Dereference the pointer.
     *
     * @return A reference to the data stored by this pointer.
     *
     * @versionadded{1.1.0}
     */
    reference operator*() noexcept { return *this->gpu_ptr_; }

    std::add_const_t<reference> operator*() const noexcept { return *this->gpu_ptr_; }

    /**
     * @brief Evaluate a method from the stored pointer.
     *
     * Since this pointer should only ever be used for arithmetic types, this is only really needed for compatibility with the smart pointer
     * requirements.
     *
     * @return The pointer stored by this object.
     *
     * @versionadded{1.1.0}
     */
    T *operator->() noexcept { return gpu_ptr_; }

    std::add_const_t<T *> operator->() const noexcept { return gpu_ptr_; }

    /**
     * @brief Increment the pointer to the next element.
     *
     * @return The next pointer.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer &operator++() noexcept {
        gpu_ptr_++;
        return gpu_ptr_;
    }

    /**
     * @brief Increment the pointer to the next element.
     *
     * @return The next pointer.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer &operator++(int) noexcept {
        gpu_ptr_++;
        return gpu_ptr_;
    }

    /**
     * @brief Decrement the pointer to the previous element.
     *
     * @return The previous pointer.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer &operator--() noexcept {
        gpu_ptr_--;
        return gpu_ptr_;
    }

    /**
     * @brief Decrement the pointer to the previous element.
     *
     * @return The previous pointer.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer &operator--(int) noexcept {
        gpu_ptr_--;
        return gpu_ptr_;
    }

    /**
     * @brief Increment the pointer several times.
     *
     * @param[in] n The number of elements to increment by.
     *
     * @return A pointer that is the given number of elements ahead.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer &operator+=(difference_type &&n) noexcept {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    /**
     * @brief Decrement the pointer several times.
     *
     * @param[in] n The number of elements to deccrement by.
     *
     * @return A pointer that is the given number of elements behind.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer &operator-=(difference_type &&n) noexcept {
        gpu_ptr_ += n;
        return gpu_ptr_;
    }

    /**
     * @brief Increment the pointer several times.
     *
     * @param[in] n The number of elements to increment by.
     *
     * @return A pointer that is the given number of elements ahead.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer operator+(difference_type &&n) const noexcept { return gpu_ptr_ + n; }

    /**
     * @brief Decrement the pointer several times.
     *
     * @param[in] n The number of elements to deccrement by.
     *
     * @return A pointer that is the given number of elements behind.
     *
     * @versionadded{1.1.0}
     */
    constexpr pointer operator-(difference_type &&n) const noexcept { return gpu_ptr_ - n; }

    /**
     * @brief Calculate the distance between two pointers.
     *
     * @param[in] other The starting point.
     *
     * @return The distance between two pointers.
     *
     * @versionadded{1.1.0}
     */
    constexpr difference_type operator-(GPUPointer<T> const &other) const noexcept { return gpu_ptr_ - other.gpu_ptr_; }

    /**
     * @brief Subscript a pointer.
     *
     * @param[in] n The offset to use for the subscript.
     *
     * @return A reference to the requested value.
     *
     * @versionadded{1.1.0}
     */
    constexpr reference operator[](difference_type &&n) noexcept { return gpu_ptr_[n]; }

    /**
     * @brief Compare two pointers.
     *
     * @param[in] other The other pointer to compare.
     *
     * @return True if this pointer is less than the other.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
    constexpr bool operator<(GPUPointer<U> const &other) const noexcept {
        return gpu_ptr_ < other.gpu_ptr_;
    }

    /**
     * @brief Compare two pointers.
     *
     * @param[in] other The other pointer to compare.
     *
     * @return True if this pointer is greater than the other.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
    constexpr bool operator>(GPUPointer<U> const &other) const noexcept {
        return gpu_ptr_ > other.gpu_ptr_;
    }

    /**
     * @brief Compare two pointers.
     *
     * @param[in] other The other pointer to compare.
     *
     * @return True if this pointer is less than or equal to the other.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
    constexpr bool operator<=(GPUPointer<U> const &other) const noexcept {
        return gpu_ptr_ <= other.gpu_ptr_;
    }

    /**
     * @brief Compare two pointers.
     *
     * @param[in] other The other pointer to compare.
     *
     * @return True if this pointer is greater than or equal to the other.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
    constexpr bool operator>=(GPUPointer<U> const &other) const noexcept {
        return gpu_ptr_ >= other.gpu_ptr_;
    }

    /**
     * @brief Get the underlying pointer.
     *
     * @return The pointer contained by this object.
     *
     * @versionadded{1.1.0}
     */
    constexpr operator T *() noexcept { return reinterpret_cast<T *>(gpu_ptr_); }

    /**
     * @brief Get the underlying pointer, but cast to a different type.
     *
     * @return The pointer contained by this object.
     *
     * @versionadded{1.1.0}
     */
    template <typename U>
        requires(!std::is_const_v<T> || std::is_const_v<U>)
    constexpr operator U *() noexcept {
        return reinterpret_cast<U *>(gpu_ptr_);
    }

    /**
     * @brief Create a pointer to a reference.
     *
     * @param[in] other The reference to wrap.
     *
     * @return The pointer object wrapping the reference.
     *
     * @versionadded{1.1.0}
     */
    static pointer pointer_to(element_type &other) noexcept { return pointer(&other); }

    static pointer pointer_to(element_type const &other) noexcept
        requires(std::is_const_v<T>)
    {
        return pointer(&other);
    }

  private:
    /**
     * @brief The pointer held by this object.
     *
     * @versionadded{1.1.0}
     */
    dev_datatype *gpu_ptr_{nullptr};

    template <typename U>
    friend void std::swap(GPUPointer<U> &, GPUPointer<U> &) noexcept;
};

} // namespace gpu

} // namespace einsums

template <typename T>
constexpr bool operator==(std::nullptr_t const &first, einsums::gpu::GPUPointer<T> const &second) noexcept {
    return second == first;
}

template <typename T>
constexpr bool operator+(ptrdiff_t &&offset, einsums::gpu::GPUPointer<T> const &base) noexcept {
    return base + offset;
}

namespace std {
/**
 * @brief Swap the contents of two pointers.
 *
 * @tparam T The type of the pointers.
 * @param[inout] a,b The pointers to swap.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
void swap(einsums::gpu::GPUPointer<T> &a, einsums::gpu::GPUPointer<T> &b) noexcept {
    auto *temp = a.gpu_ptr_;
    a.gpu_ptr_ = b.gpu_ptr_;
    b.gpu_ptr_ = a.gpu_ptr_;
}

/**
 * @brief Perform a memcpy from a host pointer to a device pointer.
 *
 * @tparam T The type of the pointers.
 * @param[out] dest The destination device pointer.
 * @param[in] src The source host pointer.
 * @param[in] bytes The number of bytes to copy.
 *
 * @return The destination pointer.
 *
 * @throws einsums::ErrorInvalidMemcpyDirection If the pointers are not in the correct location.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
einsums::gpu::GPUPointer<T> memcpy(einsums::gpu::GPUPointer<T> &dest, T const *src, size_t bytes) noexcept {
    einsums::hip_catch(hipMemcpy((void *)dest, src, bytes, hipMemcpyHostToDevice));

    return dest;
}

/**
 * @brief Perform a memcpy from a device pointer to a host pointer.
 *
 * @tparam T The type of the pointers.
 * @param[out] dest The destination host pointer.
 * @param[in] src The source device pointer.
 * @param[in] bytes The number of bytes to copy.
 *
 * @return The destination pointer.
 *
 * @throws einsums::ErrorInvalidMemcpyDirection If the pointers are not in the correct location.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
T *memcpy(T *dest, einsums::gpu::GPUPointer<T const> const &src, size_t bytes) {
    einsums::hip_catch(hipMemcpy(static_cast<void *>(dest), (void *const)src, bytes, hipMemcpyDeviceToHost));

    return dest;
}

/**
 * @brief Perform a memcpy from a device pointer to a device pointer.
 *
 * @tparam T The type of the pointers.
 * @param[out] dest The destination device pointer.
 * @param[in] src The source device pointer.
 * @param[in] bytes The number of bytes to copy.
 *
 * @return The destination pointer.
 *
 * @throws einsums::ErrorInvalidMemcpyDirection If the pointers are not in the correct location.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
einsums::gpu::GPUPointer<T> memcpy(einsums::gpu::GPUPointer<T> &dest, einsums::gpu::GPUPointer<T const> const &src, size_t bytes) {
    einsums::hip_catch(hipMemcpy((void *)dest, (void const *)src, bytes, hipMemcpyDeviceToDevice));

    return dest;
}
} // namespace std