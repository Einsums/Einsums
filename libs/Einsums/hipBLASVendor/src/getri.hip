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

auto sgetri(int n, float const *a, int lda, int *ipiv, float *c, int ldc) -> int {
    LabeledSection0();

    int info;

    float *const *a_addr_gpu = (float *const *)gpu::register_host_variable(a);
    float       **c_addr_gpu = gpu::register_host_variable(c);

    int *dev_info = gpu::register_host_variable(info);

    hipblas_catch(hipblasSgetriBatched(gpu::get_blas_handle(), n, a_addr_gpu, lda, ipiv, c_addr_gpu, ldc, dev_info, 1));

    gpu::stream_wait();
    gpu::unregister_host_variable(info);
    gpu::unregister_host_variable(a);
    gpu::unregister_host_variable(c);

    return info;
}

auto dgetri(int n, double const *a, int lda, int *ipiv, double *c, int ldc) -> int {
    LabeledSection0();

    int info;

    double *const *a_addr_gpu = (double *const *)gpu::register_host_variable(a);
    double       **c_addr_gpu = gpu::register_host_variable(c);

    int *dev_info = gpu::register_host_variable(info);

    hipblas_catch(hipblasDgetriBatched(gpu::get_blas_handle(), n, a_addr_gpu, lda, ipiv, c_addr_gpu, ldc, dev_info, 1));

    gpu::stream_wait();
    gpu::unregister_host_variable(info);
    gpu::unregister_host_variable(a);
    gpu::unregister_host_variable(c);

    return info;
}

auto cgetri(int n, std::complex<float> const *a, int lda, int *ipiv, std::complex<float> *c, int ldc) -> int {
    LabeledSection0();

    int info;

#if hipblasVersionMajor < 3
    hipblasComplex *const *a_addr_gpu = (hipblasComplex *const *)gpu::register_host_variable(a);
    hipblasComplex       **c_addr_gpu = (hipblasComplex **)gpu::register_host_variable(c);
#else
    hipComplex *const *a_addr_gpu = (hipComplex *const *)gpu::register_host_variable(a);
    hipComplex       **c_addr_gpu = (hipComplex **)gpu::register_host_variable(c);
#endif
    int *dev_info = gpu::register_host_variable(info);

    hipblas_catch(hipblasCgetriBatched(gpu::get_blas_handle(), n, a_addr_gpu, lda, ipiv, c_addr_gpu, ldc, dev_info, 1));

    gpu::stream_wait();
    gpu::unregister_host_variable(info);
    gpu::unregister_host_variable(a);
    gpu::unregister_host_variable(c);

    return info;
}

auto zgetri(int n, std::complex<double> const *a, int lda, int *ipiv, std::complex<double> *c, int ldc) -> int {
    LabeledSection0();

    int info;

#if hipblasVersionMajor < 3
    hipblasDoubleComplex *const *a_addr_gpu = (hipblasDoubleComplex *const *)gpu::register_host_variable(a);
    hipblasDoubleComplex       **c_addr_gpu = (hipblasDoubleComplex **)gpu::register_host_variable(c);
#else
    hipDoubleComplex *const *a_addr_gpu = (hipDoubleComplex *const *)gpu::register_host_variable(a);
    hipDoubleComplex       **c_addr_gpu = (hipDoubleComplex **)gpu::register_host_variable(c);
#endif

    int *dev_info = gpu::register_host_variable(info);

    hipblas_catch(hipblasZgetriBatched(gpu::get_blas_handle(), n, a_addr_gpu, lda, ipiv, c_addr_gpu, ldc, dev_info, 1));

    gpu::stream_wait();
    gpu::unregister_host_variable(info);
    gpu::unregister_host_variable(a);
    gpu::unregister_host_variable(c);

    return info;
}

} // namespace einsums::blas::hip