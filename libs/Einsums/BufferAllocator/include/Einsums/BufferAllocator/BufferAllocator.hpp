//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <complex>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <source_location>
#include <type_traits>

namespace einsums {

/**
 * @struct BufferAllocator
 *
 * @brief Allocator whose maximum size can be restricted by runtime variables.
 *
 * Not for use in vectors.
 */
template <typename T>
struct BufferAllocator {
  public:
    using pointer            = T *;
    using const_pointer      = T const *;
    using void_pointer       = void *;
    using const_void_pointer = void const *;
    using value_type         = T;
    using size_type          = size_t;
    using difference_type    = ptrdiff_t;
    using is_always_equal    = std::true_type;

    pointer allocate(size_t n) {
        pointer out;

        auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();

        if (!vars.request_bytes(n * sizeof(T))) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not allocate enough memory for buffers. Requested {} bytes.", n * sizeof(T));
        }

        out = static_cast<pointer>(malloc(n * sizeof(T)));

        if (out == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not allocate enough memory for buffers. Requested {} bytes.", n * sizeof(T));
        }

        return out;
    }

    void deallocate(pointer p, size_t n) {
        auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();

        vars.release_bytes(n * sizeof(T));

        free(static_cast<void *>(p));
    }

    size_type max_size() const { return detail::Einsums_BufferAllocator_vars::get_singleton().get_max_size() / sizeof(T); }

    bool operator==(BufferAllocator<T> const &other) const { return true; }
    bool operator!=(BufferAllocator<T> const &other) const { return false; }
};

} // namespace einsums