//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {

#define GEMV_CHECK(transa, m, n, lda, incx, incy)                                                                                          \
    char ta = std::tolower(transa);                                                                                                        \
                                                                                                                                           \
    if (ta != 'n' && ta != 'c' && ta != 't') {                                                                                             \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,                                                                                     \
                                "The first argument (transa) to gemv call is invalid. Expected n, t, or c, case insensitive, got {}.",     \
                                transa);                                                                                                   \
    }                                                                                                                                      \
                                                                                                                                           \
    if ((m) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The second argument (m) to gemv call is invalid. Expected a number greater than or equal to zero, got {}.", m);               \
    }                                                                                                                                      \
    if ((n) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error, "The third argument (n) to gemv call is invalid. Expected a number greater than or equal to zero, got {}.", \
            n);                                                                                                                            \
    }                                                                                                                                      \
    if ((lda) < std::max((int)1, m)) {                                                                                                   \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The sixth argument (lda) to gemv call is invalid. Expected a number at least 1 and at least {}, got {}.", \
                                m, lda);                                                                                                   \
    }                                                                                                                                      \
    if ((incx) == 0) {                                                                                                                     \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The eighth argument (incx) to gemv call is invalid. Expected a non-zero value, got zero.");               \
    }                                                                                                                                      \
    if ((incy) == 0) {                                                                                                                     \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The eleventh argument (incy) to gemv call is invalid. Expected a non-zero value, got zero.");             \
    }

void sgemv(char transa, int m, int n, float alpha, float const *a, int lda, float const *x, int incx, float beta, float *y,
           int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    float *alpha_gpu = gpu::register_host_variable(alpha), *beta_gpu = gpu::register_host_variable(beta);

    hipblas_catch(hipblasSgemv(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, alpha_gpu, a, lda, x, incx, beta_gpu, y, incy));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

void dgemv(char transa, int m, int n, double alpha, double const *a, int lda, double const *x, int incx, double beta, double *y,
           int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    double *alpha_gpu = gpu::register_host_variable(alpha), *beta_gpu = gpu::register_host_variable(beta);

    hipblas_catch(hipblasDgemv(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, alpha_gpu, a, lda, x, incx, beta_gpu, y, incy));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

void cgemv(char transa, int m, int n, std::complex<float> alpha, std::complex<float> const *a, int lda, std::complex<float> const *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    hipblasComplex *alpha_gpu = (hipblasComplex *)gpu::register_host_variable(alpha),
                   *beta_gpu  = (hipblasComplex *)gpu::register_host_variable(beta);

    hipblas_catch(hipblasCgemv(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, alpha_gpu, (hipblasComplex *)a, lda,
                               (hipblasComplex *)x, incx, beta_gpu, (hipblasComplex *)y, incy));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

void zgemv(char transa, int m, int n, std::complex<double> alpha, std::complex<double> const *a, int lda,
           std::complex<double> const *x, int incx, std::complex<double> beta, std::complex<double> *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    hipblasDoubleComplex *alpha_gpu = (hipblasDoubleComplex *)gpu::register_host_variable(alpha),
                         *beta_gpu  = (hipblasDoubleComplex *)gpu::register_host_variable(beta);

    hipblas_catch(hipblasZgemv(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, alpha_gpu, (hipblasDoubleComplex *)a, lda,
                               (hipblasDoubleComplex *)x, incx, beta_gpu, (hipblasDoubleComplex *)y, incy));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

} // namespace einsums::blas::hip