//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPUMemory/GPUPointer.hpp>
#include <Einsums/GPUMemory/InitModule.hpp>
#include <Einsums/GPUMemory/ModuleVars.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <complex>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <source_location>
#include <type_traits>

#include "Einsums/TypeSupport/SizeOf.hpp"

namespace einsums {

namespace gpu {

template <typename T>
struct GPUAllocator {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if @c T is complex or @c void.
     *
     * When @c T is complex, this will give the translation from the C++ @c std::complex type to the appropriate
     * HIP complex type. When @c T is void, this will give @c uint8_t in order to give good sizes.
     */
    using dev_datatype = SwitchT<std::remove_cv_t<T>, Case<uint8_t, void>, Case<hipFloatComplex, std::complex<float>>,
                                 Case<hipDoubleComplex, std::complex<double>>, Default<T>>;

    /**
     * @typedef host_datatype
     *
     * @brief Exposes the template parameter to the programmer while handling @c void cases.
     *
     * When @c T is void, this will give @c uint8_t in order to give good sizes. Otherwise, it
     * will simply be @c T with const and volatile removed.
     */
    using host_datatype = std::conditional_t<std::is_void_v<std::remove_cv_t<T>>, uint8_t, std::remove_cv_t<T>>;

    /**
     * @typedef pointer
     *
     * @brief The kinds of pointers returned by this allocator.
     *
     * This allocator returns GPUPointer<T>, which are smart pointers that wrap pointers on the GPU.
     */
    using pointer = GPUPointer<T>;

    /**
     * @typedef const_pointer
     *
     * @brief The kinds of pointers returned by this allocator but const.
     *
     * This allocator returns a const form of GPUPointer<T>, which are smart pointers that wrap pointers on the GPU.
     */
    using const_pointer = GPUPointer<std::add_const_t<T>>;

    /**
     * @typedef void_pointer
     *
     * @brief The kinds of pointers returned by this allocator but typeless.
     *
     * This allocator returns GPUPointer<void> or GPUPointer<void const>, which are smart pointers that wrap pointers on the GPU.
     */
    using void_pointer = std::conditional_t<std::is_const_v<T>, GPUPointer<void const>, GPUPointer<void>>;

    /**
     * @typedef const_void_pointer
     *
     * @brief The kinds of pointers returned by this allocator but typeless.
     *
     * This allocator returns GPUPointer<void const>, which are smart pointers that wrap pointers on the GPU.
     */
    using const_void_pointer = GPUPointer<void const>;

    /**
     * @typedef value_type
     *
     * @brief The type of data stored by the pointers.
     *
     * This is the data type on the device, not the host. Complex values are transformed to the device equivalents.
     */
    using value_type = dev_datatype;

    /**
     * @typedef size_type
     *
     * @brief The type representing the sizes of pointers.
     *
     * It's @c size_t .
     */
    using size_type = size_t;

    /**
     * @typedef difference_type
     *
     * @brief The type representing differences between pointers.
     *
     * It's @c ptrdiff_t .
     */
    using difference_type = ptrdiff_t;

    /**
     * @property itemsize
     *
     * @brief The size of an individual element.
     *
     * This is provided since @c sizeof(void) is not a valid statement, but we still need to find the size of
     * elements contained in void pointers, which is 1 byte. When the data type is not void, then this just
     * gives the size of that data type.
     */
    constexpr static size_t itemsize = SizeOfV<T>;

    /**
     * @brief Allocate a number of elements.
     *
     * The total allocation size in bytes will be equal to the number of requested elements times the size in bytes of the element type.
     * If the data type is void, then the number of bytes will be the number of elements.
     *
     * @param n The number of elements to allocate.
     *
     * @return A pointer to the allocated memory.
     *
     * @throws std::runtime_errror if the allocator can not reserve enough memory.
     */
    pointer allocate(size_t n) {
        T *out;

        auto &vars = detail::Einsums_GPUMemory_vars::get_singleton();

        if (!vars.try_allocate(n * itemsize)) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not allocate enough memory on the GPU device. Requested {} bytes.",
                                    n * itemsize);
        }

        hip_catch(hipMalloc((void **)&out, n * itemsize));

        return pointer(out);
    }

    /**
     * @brief Try to allocate a number of elements.
     *
     * The total allocation size in bytes will be equal to the number of requested elements times the size in bytes of the element type.
     * If the data type is void, then the number of bytes will be the number of elements.
     *
     * @param n The number of elements to allocate.
     *
     * @return A pointer to the allocated memory. If memory could not be allocated, this will be the null pointer.
     */
    pointer try_allocate(size_t n) {
        T *out;

        auto &vars = detail::Einsums_GPUMemory_vars::get_singleton();

        if (!vars.try_allocate(n * itemsize)) {
            return pointer(nullptr);
        }

        hip_catch(hipMalloc((void **)&out, n * itemsize));

        return pointer(out);
    }

    /**
     * @brief Deallocate a pointer that points to a number of elements.
     *
     * @param p The pointer to deallocate.
     * @param n The number of elements the pointer pointed to.
     */
    void deallocate(pointer p, size_t n) {
        auto &vars = detail::Einsums_GPUMemory_vars::get_singleton();

        vars.deallocate(n * itemsize);

        hip_catch(hipFree((void *)p));
    }

    /**
     * @brief Get the max allocation size in elements.
     *
     * This is based on the command line arguments passed into Einsums.
     */
    size_type max_size() const { return detail::Einsums_GPUMemory_vars::get_singleton().get_max_size() / itemsize; }
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
};

} // namespace gpu

} // namespace einsums