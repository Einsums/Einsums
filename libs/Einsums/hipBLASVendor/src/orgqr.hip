//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <exception>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {

#define ORGQR(Type, lc, uc)                                                                                                                \
    auto lc##orgqr(int m, int n, int k, Type *a, int lda, Type *tau) -> int {                                                              \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int info{0};                                                                                                                       \
        int lwork{-1};                                                                                                                     \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("orgqr warning: lda < n, lda = {}, n = {}", lda, n);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        hipsolver_catch(hipsolver##uc##orgqr_bufferSize(gpu::get_solver_handle(), m, n, k, a, lda, tau, &lwork));                          \
        gpu::stream_wait();                                                                                                                \
                                                                                                                                           \
        auto alloc = gpu::GPUAllocator<Type>();                                                                                            \
        auto work  = alloc.allocate(lwork);                                                                                                \
                                                                                                                                           \
        int *dev_info = gpu::register_host_variable(info);                                                                                 \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        try {                                                                                                                              \
            hipsolver_catch(hipsolver##uc##orgqr(gpu::get_solver_handle(), m, n, k, a, lda, tau, work, lwork, dev_info));                  \
            gpu::stream_wait();                                                                                                            \
        } catch (std::exception &) {                                                                                                       \
            gpu::unregister_host_variable(info);                                                                                           \
            alloc.deallocate(work, lwork);                                                                                                 \
            std::rethrow_exception(std::current_exception());                                                                              \
        }                                                                                                                                  \
        gpu::unregister_host_variable(info);                                                                                               \
        alloc.deallocate(work, lwork);                                                                                                     \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

ORGQR(double, d, D);
ORGQR(float, s, S);

#define UNGQR(Type1, Type2, lc, uc)                                                                                                        \
    auto lc##ungqr(int m, int n, int k, Type1 *a, int lda, Type1 *tau) -> int {                                                            \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int info{0};                                                                                                                       \
        int lwork{-1};                                                                                                                     \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            EINSUMS_LOG_WARN("ungqr warning: lda < n, lda = {}, n = {}", lda, n);                                                          \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        hipsolver_catch(hipsolver##uc##ungqr_bufferSize(gpu::get_solver_handle(), m, n, k, (Type2 *)a, lda, (Type2 *)tau, &lwork));        \
        gpu::stream_wait();                                                                                                                \
                                                                                                                                           \
        auto alloc = gpu::GPUAllocator<Type1>();                                                                                           \
        auto work  = alloc.allocate(lwork);                                                                                                \
                                                                                                                                           \
        int *dev_info = gpu::register_host_variable(info);                                                                                 \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        try {                                                                                                                              \
            hipsolver_catch(                                                                                                               \
                hipsolver##uc##ungqr(gpu::get_solver_handle(), m, n, k, (Type2 *)a, lda, (Type2 *)tau, work, lwork, dev_info));            \
            gpu::stream_wait();                                                                                                            \
        } catch (std::exception &) {                                                                                                       \
            gpu::unregister_host_variable(info);                                                                                           \
            alloc.deallocate(work, lwork);                                                                                                 \
            std::rethrow_exception(std::current_exception());                                                                              \
        }                                                                                                                                  \
        gpu::unregister_host_variable(info);                                                                                               \
        alloc.deallocate(work, lwork);                                                                                                     \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

UNGQR(std::complex<float>, hipComplex, c, C);
UNGQR(std::complex<double>, hipDoubleComplex, z, Z);

} // namespace einsums::blas::hip