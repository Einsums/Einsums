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

#define GESVD(Type1, Type2, lcletter, UCLETTER)                                                                                            \
    auto lcletter##gesvd(char jobu, char jobvt, int m, int n, Type1 *a, int lda, Type1 *s, Type1 *u, int ldu, Type1 *vt, int ldvt,         \
                         Type1 *superb) -> int {                                                                                           \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int info  = 0;                                                                                                                     \
        int lwork = -1;                                                                                                                    \
                                                                                                                                           \
        int i;                                                                                                                             \
                                                                                                                                           \
        int ncols_u  = vendor::lsame(jobu, 'a') ? m : (vendor::lsame(jobu, 's') ? std::min(m, n) : 1);                                     \
        int ncols_vt = (vendor::lsame(jobvt, 'a')) ? n : (vendor::lsame(jobvt, 's') ? std::min(m, n) : 1);                                 \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("gesvd warning: lda < m, lda = {}, m = {}", lda, m);                                                          \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            EINSUMS_LOG_WARN("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                        \
        }                                                                                                                                  \
        if (ldvt < ncols_vt) {                                                                                                             \
            EINSUMS_LOG_WARN("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) size */                                                                                          \
        hipsolver_catch(hipsolver##UCLETTER##gesvd_bufferSize(gpu::get_solver_handle(), jobu, jobvt, m, n, &lwork));                       \
        gpu::stream_wait();                                                                                                                \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        auto alloc = gpu::GPUAllocator<Type1>();                                                                                           \
                                                                                                                                           \
        auto work  = alloc.allocate(lwork);                                                                                                \
        auto rwork = alloc.allocate(5 * std::min(m, n));                                                                                   \
                                                                                                                                           \
        int *dev_info = gpu::register_host_variable(info);                                                                                 \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        try {                                                                                                                              \
            hipsolver_catch(hipsolver##UCLETTER##gesvd(gpu::get_solver_handle(), jobu, jobvt, m, n, (Type2 *)a, lda, (Type1 *)s,           \
                                                       (Type2 *)u, ldu, (Type2 *)vt, ldvt, (Type2 *)work, lwork, (Type1 *)rwork,           \
                                                       dev_info));                                                                         \
            gpu::stream_wait();                                                                                                            \
        } catch (std::exception &) {                                                                                                       \
            gpu::unregister_host_variable(info);                                                                                           \
            alloc.deallocate(work, lwork);                                                                                                 \
            alloc.deallocate(rwork, 5 * std::min(m, n));                                                                                   \
            std::rethrow_exception(std::current_exception());                                                                              \
        }                                                                                                                                  \
        gpu::unregister_host_variable(info);                                                                                               \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Backup significant data from working arrays into superb */                                                                      \
        for (i = 0; i < std::min(m, n) - 1; i++) {                                                                                         \
            superb[i] = work[i + 1];                                                                                                       \
        }                                                                                                                                  \
        alloc.deallocate(work, lwork);                                                                                                     \
        alloc.deallocate(rwork, 5 * std::min(m, n));                                                                                       \
                                                                                                                                           \
        return 0;                                                                                                                          \
    }

GESVD(double, double, d, D);
GESVD(float, float, s, S);

#define GESVD_complex(Type1, Type2, lcletter, UCLETTER)                                                                                    \
    auto lcletter##gesvd(char jobu, char jobvt, int m, int n, std::complex<Type1> *a, int lda, Type1 *s, std::complex<Type1> *u, int ldu,  \
                         std::complex<Type1> *vt, int ldvt, std::complex<Type1> *superb) -> int {                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int info  = 0;                                                                                                                     \
        int lwork = -1;                                                                                                                    \
                                                                                                                                           \
        int i;                                                                                                                             \
                                                                                                                                           \
        int ncols_u  = vendor::lsame(jobu, 'a') ? m : (vendor::lsame(jobu, 's') ? std::min(m, n) : 1);                                     \
        int ncols_vt = (vendor::lsame(jobvt, 'a')) ? n : (vendor::lsame(jobvt, 's') ? std::min(m, n) : 1);                                 \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            EINSUMS_LOG_WARN("gesvd warning: lda < m, lda = {}, m = {}", lda, m);                                                          \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            EINSUMS_LOG_WARN("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                        \
        }                                                                                                                                  \
        if (ldvt < ncols_vt) {                                                                                                             \
            EINSUMS_LOG_WARN("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) size */                                                                                          \
        hipsolver_catch(hipsolver##UCLETTER##gesvd_bufferSize(gpu::get_solver_handle(), jobu, jobvt, m, n, &lwork));                       \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        auto alloc  = gpu::GPUAllocator<Type1>();                                                                                          \
        auto ralloc = gpu::GPUAllocator<Type2>();                                                                                          \
                                                                                                                                           \
        auto work  = alloc.allocate(lwork);                                                                                                \
        auto rwork = ralloc.allocate(5 * std::min(m, n));                                                                                  \
                                                                                                                                           \
        int *dev_info = gpu::register_host_variable(info);                                                                                 \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        try {                                                                                                                              \
            hipsolver_catch(hipsolver##UCLETTER##gesvd(gpu::get_solver_handle(), jobu, jobvt, m, n, (Type2 *)a, lda, (Type1 *)s,           \
                                                       (Type2 *)u, ldu, (Type2 *)vt, ldvt, (Type2 *)work, lwork, (Type1 *)rwork,           \
                                                       dev_info));                                                                         \
            gpu::stream_wait();                                                                                                            \
        } catch (std::exception &) {                                                                                                       \
            gpu::unregister_host_variable(info);                                                                                           \
            alloc.deallocate(work, lwork);                                                                                                 \
            ralloc.deallocate(rwork, 5 * std::min(m, n));                                                                                  \
            std::rethrow_exception(std::current_exception());                                                                              \
        }                                                                                                                                  \
                                                                                                                                           \
        gpu::unregister_host_variable(info);                                                                                               \
                                                                                                                                           \
        if (info != 0) {                                                                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Backup significant data from working arrays into superb */                                                                      \
        for (i = 0; i < std::min(m, n) - 1; i++) {                                                                                         \
            superb[i] = work[i + 1];                                                                                                       \
        }                                                                                                                                  \
        alloc.deallocate(work, lwork);                                                                                                     \
        ralloc.deallocate(rwork, 5 * std::min(m, n));                                                                                      \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESVD_complex(double, hipDoubleComplex, z, Z);
GESVD_complex(float, hipComplex, c, C);

} // namespace einsums::blas::hip