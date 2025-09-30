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

auto sgetrf(int m, int n, float *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    int lwork;

    hipsolver_catch(hipsolverSgetrf_bufferSize(gpu::get_solver_handle(), m, n, a, lda, &lwork));

    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<float>();

    auto work = alloc.allocate(lwork);

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverSgetrf(gpu::get_solver_handle(), m, n, a, lda, work, lwork, ipiv, dev_info));
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

auto dgetrf(int m, int n, double *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    int lwork;

    hipsolver_catch(hipsolverDgetrf_bufferSize(gpu::get_solver_handle(), m, n, a, lda, &lwork));

    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<double>();

    auto work = alloc.allocate(lwork);

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverDgetrf(gpu::get_solver_handle(), m, n, a, lda, work, lwork, ipiv, dev_info));
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

auto cgetrf(int m, int n, std::complex<float> *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    int lwork;

    hipsolver_catch(hipsolverCgetrf_bufferSize(gpu::get_solver_handle(), m, n, (hipComplex *)a, lda, &lwork));

    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<std::complex<float>>();

    auto work = alloc.allocate(lwork);

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverCgetrf(gpu::get_solver_handle(), m, n, (hipComplex *)a, lda, work, lwork, ipiv, dev_info));
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

auto zgetrf(int m, int n, std::complex<double> *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    int lwork;

    hipsolver_catch(hipsolverZgetrf_bufferSize(gpu::get_solver_handle(), m, n, (hipDoubleComplex *)a, lda, &lwork));

    gpu::stream_wait();

    auto alloc = gpu::GPUAllocator<std::complex<double>>();

    auto work = alloc.allocate(lwork);

    int *dev_info = gpu::register_host_variable(info);

    try {
        hipsolver_catch(hipsolverZgetrf(gpu::get_solver_handle(), m, n, (hipDoubleComplex *)a, lda, work, lwork, ipiv, dev_info));
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