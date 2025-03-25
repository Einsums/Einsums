//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPUMemory/InitModule.hpp>
#include <Einsums/GPUMemory/ModuleVars.hpp>

#include <complex>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <source_location>
#include <type_traits>

#include "Einsums/GPUMemory/GPUPointer.hpp"
#include "Einsums/StringUtil/MemoryString.hpp"

namespace einsums {

namespace gpu {

template <typename T>
struct GPUAllocator {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype = std::conditional_t<
        std::is_same_v<std::remove_cv_t<T>, std::complex<float>>, hipFloatComplex,
        std::conditional_t<std::is_same_v<std::remove_cv_t<T>, std::complex<double>>, hipDoubleComplex, std::remove_cv_t<T>>>;
    using pointer            = GPUPointer<T>;
    using const_pointer      = GPUPointer<T const>;
    using void_pointer       = GPUPointer<void>;
    using const_void_pointer = GPUPointer<void const>;
    using value_type         = dev_datatype;
    using size_type          = size_t;
    using difference_type    = ptrdiff_t;

    pointer allocate(size_t n) {
        dev_datatype *out;

        auto &vars = detail::Einsums_GPUMemory_vars::get_singleton();

        if (!vars.try_allocate(n * sizeof(T))) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not allocate enough memory on the GPU device. Requested {} bytes.",
                                    n * sizeof(T));
        }

        hip_catch(hipMalloc((void **)&out, n * sizeof(T)));

        return pointer(out);
    }

    void deallocate(pointer p, size_t n) {
        auto &vars = detail::Einsums_GPUMemory_vars::get_singleton();

        vars.deallocate(n * sizeof(T));

        hip_catch(hipFree((void *)p));
    }

    void construct(pointer xp, T const &value) {
        hip_catch(hipMemcpy((void *)xp, (void const *)&value, sizeof(value), hipMemcpyHostToDevice));
    }

    void destroy(pointer xp) {
        ; // Do nothing.
    }

    size_type max_size() const { return detail::Einsums_GPUMemory_vars::get_singleton().get_max_size() / sizeof(T); }
};

template <typename T>
struct MappedAllocator {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype = std::conditional_t<
        std::is_same_v<std::remove_cv_t<T>, std::complex<float>>, hipFloatComplex,
        std::conditional_t<std::is_same_v<std::remove_cv_t<T>, std::complex<double>>, hipDoubleComplex, std::remove_cv_t<T>>>;
    using pointer            = T *;
    using const_pointer      = T const *;
    using void_pointer       = void *;
    using const_void_pointer = void const *;
    using value_type         = T;
    using size_type          = size_t;
    using difference_type    = ptrdiff_t;

    pointer allocate(size_t n) {
        pointer host_ptr = new T[n];

        hip_catch(hipHostRegister(host_ptr, n * sizeof(T), hipHostRegisterDefault));

        return host_ptr;
    }

    void deallocate(pointer p, size_t n) {
        hip_catch(hipHostUnregister(static_cast<void *>(p)));
        delete[] p;
    }

    template <typename... Args>
    void construct(pointer xp, Args &&...args) {
        *xp = T(std::forward<Args>(args)...);
    }
};

} // namespace gpu

} // namespace einsums