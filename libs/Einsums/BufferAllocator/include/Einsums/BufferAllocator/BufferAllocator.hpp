//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <complex>
#include <source_location>
#include <type_traits>

namespace einsums {

/**
 * @struct BufferAllocator
 *
 * @brief Allocator whose maximum size can be restricted by runtime variables.
 *
 * The maximum size is controlled by the global configuration options. In particular, the option to control the size is
 * @c --einsums:buffer-size and the string in the configuration mapping is @c buffer-size with the standard variants,
 * such as @c BUFFER_SIZE .
 *
 * Use in dynamic allocations with frequent resizing is discouraged, such as in a vector. Resizing requires the buffer
 * to be reallocated while still being allocated. If the buffers are big when resized, this means that the buffer
 * allocation will fail even though the result would be expected to fall within the range of acceptable sizes.
 * However, when using this to allocate a buffer once, this should be fine.
 *
 * This allocator follows the C++ standard for allocators, and can be used in templates that use them, such as
 * containers, smart pointers, and more.
 */
template <typename T>
struct BufferAllocator {
  public:
    /**
     * @typedef pointer
     *
     * Represents the type of pointer handled by this allocator.
     */
    using pointer = T *;

    /**
     * @typedef const_pointer
     *
     * Represents the type of pointer handled by this allocator, but for const operations.
     */
    using const_pointer = T const *;

    /**
     * @typedef void_pointer
     *
     * Represents the type of typeless pointer handled by this allocator.
     */
    using void_pointer = void *;

    /**
     * @typedef const_void_pointer
     *
     * Represents the type of typeless pointer handled by this allocator, but for const operations.
     */
    using const_void_pointer = void const *;

    /**
     * @typedef value_type
     *
     * The type of buffers this allocator makes.
     */
    using value_type = T;

    /**
     * @typedef size_type
     *
     * The type used to represent sizes for this buffer.
     */
    using size_type = size_t;

    /**
     * @typedef difference_type
     *
     * The type used to represent address offsets.
     */
    using difference_type = ptrdiff_t;

    /**
     * @typedef is_always_equal
     *
     * Flag that indicates that all allocators of this type are considered to be equal.
     */
    using is_always_equal = std::true_type;

    /**
     * @property type_size
     *
     * The size of the types handled by this allocator. This is needed to handle void pointers, which traditionally
     * don't have a size, but should be treated as one byte.
     */
    constexpr static size_t type_size = sizeof(std::conditional_t<std::is_void_v<std::remove_cv_t<T>>, char, T>);

    /**
     * @brief Allocate an array of values.
     *
     * This does not initialize the values in the array. If given zero, this will return the null pointer.
     *
     * @param n The number of elements in the array.
     *
     * @return A pointer to the newly allocated memory.
     *
     * @throws std::runtime_error When the allocation size is too large or the allocation returns an unexpected null pointer.
     */
    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        pointer out;

        if (!reserve(n)) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "Could not allocate enough memory for buffers. Requested {} elements or {} bytes, but only {} bytes "
                                    "available out of {} bytes maximum.",
                                    n, n * type_size, available_size(), max_size());
        }

        out = static_cast<pointer>(malloc(n * type_size));

        if (out == nullptr) {
            EINSUMS_THROW_EXCEPTION(
                std::runtime_error,
                "Could not allocate enough memory for buffers. Requested {} elements or {} bytes, but malloc returned a null pointer.", n,
                n * type_size);
        }

        return out;
    }

    /**
     * @brief Reserve a number of elements without allocating.
     *
     * This function is used when you want to allocate some memory with some other allocator, such as the
     * default allocator, but still want to track and limit the memory being used.
     *
     * @param n The number of elements to reserve.
     *
     * @return True if the allocation will be successful, false if it will fail.
     */
    [[nodiscard("This function tells you when an allocation fails due to being out of memory. Don't ignore its return value. It is bad "
                "form.")]] bool
    reserve(size_type n) {
        if (n == 0) {
            return true;
        }

        auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();

        return vars.request_bytes(n * type_size);
    }

    /**
     * @brief Release a number of elements without freeing.
     *
     * This function is used when you want to deallocate some memory with some other allocator, such as the
     * default allocator, but still want to track the memory being used.
     *
     * @param n The number of elements to release.
     */
    void release(size_type n) {
        if (n == 0) {
            return;
        }

        auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();

        vars.release_bytes(n * type_size);
    }

    /**
     * @brief Deallocate a number of elements.
     *
     * @param p The pointer to free.
     * @param n The number of elements the pointer points to.
     */
    void deallocate(pointer p, size_type n) {
        release(n);

        if (p != nullptr) {
            free(static_cast<void *>(p));
        }
    }

    /**
     * @brief Query the maximum number of elements the allocator can accept.
     *
     * This will return the number of elements as determined by the global buffer size limit.
     *
     * @return The number of elements specified by the buffer size option.
     */
    size_type max_size() const { return detail::Einsums_BufferAllocator_vars::get_singleton().get_max_size() / type_size; }

    /**
     * @brief Query the number of elements the allocator has free.
     *
     * This will return the number of elements that have not yet been allocated.
     *
     * @return The number of elements available to allocate.
     */
    size_type available_size() const { return detail::Einsums_BufferAllocator_vars::get_singleton().get_available() / type_size; }

    /**
     * @brief Test whether two buffer allocators are the same.
     *
     * All buffer allocators are considered to be the same, so this will always return true.
     *
     * @param other The allocator to compare to.
     * @return Always returns true.
     */
    constexpr bool operator==(BufferAllocator<T> const &other) const { return true; }

    /**
     * @brief Test whether two buffer allocators are not the same.
     *
     * All buffer allocators are considered to be the same, so this will always return false.
     *
     * @param other The allocator to compare to.
     * @return Always returns false.
     */
    constexpr bool operator!=(BufferAllocator<T> const &other) const { return false; }
};

#ifndef WINDOWS

extern template struct EINSUMS_EXPORT BufferAllocator<void>;
extern template struct EINSUMS_EXPORT BufferAllocator<float>;
extern template struct EINSUMS_EXPORT BufferAllocator<double>;
extern template struct EINSUMS_EXPORT BufferAllocator<std::complex<float>>;
extern template struct EINSUMS_EXPORT BufferAllocator<std::complex<double>>;

#endif

} // namespace einsums