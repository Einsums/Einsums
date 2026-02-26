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

#define GEQRF(Type1, Type2, lc, uc)                                                                                                        \
    auto lc##geqrf(int m, int n, Type1 *a, int lda, Type1 *tau) -> int {                                                                   \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int lwork{-1};                                                                                                                     \
        int info;                                                                                                                          \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        hipsolver_catch(hipsolver##uc##geqrf_bufferSize(gpu::get_solver_handle(), m, n, (Type2 *)a, lda, &lwork));                         \
        gpu::stream_wait();                                                                                                                \
                                                                                                                                           \
        auto alloc = gpu::GPUAllocator<Type1>();                                                                                           \
        auto work  = alloc.allocate(lwork);                                                                                                \
                                                                                                                                           \
        int *dev_info = gpu::register_host_variable(info);                                                                                 \
        /* Call LAPACK function and adjust info */                                                                                         \
        try {                                                                                                                              \
            hipsolver_catch(                                                                                                               \
                hipsolver##uc##geqrf(gpu::get_solver_handle(), m, n, (Type2 *)a, lda, (Type2 *)tau, (Type2 *)work, lwork, dev_info));      \
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

GEQRF(double, double, d, D);
GEQRF(float, float, s, S);
GEQRF(std::complex<double>, hipDoubleComplex, z, Z);
GEQRF(std::complex<float>, hipComplex, c, C);

} // namespace einsums::blas::hip