//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <complex>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>
#include <Einsums/Errors/Error.hpp>

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
    using dev_datatype       = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipFloatComplex,
                                                  std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;
    using pointer            = dev_datatype *;
    using const_pointer      = dev_datatype const *;
    using void_pointer       = void *;
    using const_void_pointer = void const *;
    using value_type         = dev_datatype;
    using size_type          = size_t;
    using difference_type    = ptrdiff_t;

    pointer allocate(size_t n) {
        pointer out;

        hip_catch(hipMalloc((void **)&out, n * sizeof(T)));

        return out;
    }

    void deallocate(pointer p, size_t n) { hip_catch(hipFree(static_cast<void *>(p))); }

    void construct(pointer xp, T const &value) {
        hip_catch(hipMemcpy((void *)xp, (void const *)&value, sizeof(value), hipMemcpyHostToDevice));
    }
};

template <typename T>
struct MappedAllocator {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype       = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipFloatComplex,
                                                  std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;
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

    template<typename... Args>
    void construct(pointer xp, Args&&... args) {
        *xp = T(std::forward<Args>(args)...);
    }
};

} // namespace gpu

} // namespace einsums