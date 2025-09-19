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

auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, float *x, int ldx) -> int {
    LabeledSection0();

    size_t lwork;

    hipsolver_catch(hipsolverSSgesv_bufferSize(gpu::get_solver_handle(), n, nrhs, a, lda, ipiv, b, ldb, x, ldx, &lwork));
    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<void>();

    auto work = alloc.allocate(lwork);
    int  iters;
    int  info;

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverSSgesv(gpu::get_solver_handle(), n, nrhs, a, lda, ipiv, b, ldb, x, ldx, work, lwork, &iters, dev_info));
        gpu::stream_wait();
    } catch (std::exception &) {
        gpu::unregister_host_variable(info);
        alloc.deallocate(work, lwork);
        std::rethrow_exception(std::current_exception());
    }
    gpu::unregister_host_variable(info);
    alloc.deallocate(work, lwork);

    return info;
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, double *x, int ldx) -> int {
    LabeledSection0();

    size_t lwork;

    hipsolver_catch(hipsolverDDgesv_bufferSize(gpu::get_solver_handle(), n, nrhs, a, lda, ipiv, b, ldb, x, ldx, &lwork));
    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<void>();

    auto work = alloc.allocate(lwork);
    int  iters;
    int  info;

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverDDgesv(gpu::get_solver_handle(), n, nrhs, a, lda, ipiv, b, ldb, x, ldx, work, lwork, &iters, dev_info));
        gpu::stream_wait();
    } catch (std::exception &) {
        gpu::unregister_host_variable(info);
        alloc.deallocate(work, lwork);
        std::rethrow_exception(std::current_exception());
    }
    gpu::unregister_host_variable(info);
    alloc.deallocate(work, lwork);

    return info;
}

auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb, std::complex<float> *x, int ldx)
    -> int {
    LabeledSection0();

    size_t lwork;

    hipsolver_catch(hipsolverCCgesv_bufferSize(gpu::get_solver_handle(), n, nrhs, (hipComplex *)a, lda, ipiv, (hipComplex *)b, ldb,
                                               (hipComplex *)x, ldx, &lwork));
    gpu::stream_wait();
    auto alloc = gpu::GPUAllocator<void>();

    auto work = alloc.allocate(lwork);
    int  iters;
    int  info;

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverCCgesv(gpu::get_solver_handle(), n, nrhs, (hipComplex *)a, lda, ipiv, (hipComplex *)b, ldb,
                                        (hipComplex *)x, ldx, work, lwork, &iters, dev_info));
        gpu::stream_wait();
    } catch (std::exception &) {
        gpu::unregister_host_variable(info);
        alloc.deallocate(work, lwork);
        std::rethrow_exception(std::current_exception());
    }
    gpu::unregister_host_variable(info);
    alloc.deallocate(work, lwork);

    return info;
}

auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb, std::complex<double> *x, int ldx)
    -> int {
    LabeledSection0();

    size_t lwork;

    hipsolver_catch(hipsolverZZgesv_bufferSize(gpu::get_solver_handle(), n, nrhs, (hipDoubleComplex *)a, lda, ipiv, (hipDoubleComplex *)b,
                                               ldb, (hipDoubleComplex *)x, ldx, &lwork));
    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<void>();

    auto work = alloc.allocate(lwork);
    int  iters;
    int  info;

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverZZgesv(gpu::get_solver_handle(), n, nrhs, (hipDoubleComplex *)a, lda, ipiv, (hipDoubleComplex *)b, ldb,
                                        (hipDoubleComplex *)x, ldx, work, lwork, &iters, dev_info));
        gpu::stream_wait();
    } catch (std::exception &) {
        gpu::unregister_host_variable(info);
        alloc.deallocate(work, lwork);
        std::rethrow_exception(std::current_exception());
    }
    gpu::unregister_host_variable(info);
    alloc.deallocate(work, lwork);

    return info;
}

} // namespace einsums::blas::hip