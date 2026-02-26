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
    if ((lda) < std::max((int64_t)1, m)) {                                                                                                     \
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

void sgemv(char transa, int64_t m, int64_t n, float alpha, float const *a, int64_t lda, float const *x, int64_t incx, float beta, float *y, int64_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    hipblas_catch(hipblasSgemv_64(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, &alpha, a, lda, x, incx, &beta, y, incy));

    gpu::stream_wait();
}

void dgemv(char transa, int64_t m, int64_t n, double alpha, double const *a, int64_t lda, double const *x, int64_t incx, double beta, double *y, int64_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    hipblas_catch(hipblasDgemv_64(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, &alpha, a, lda, x, incx, &beta, y, incy));

    gpu::stream_wait();
}

void cgemv(char transa, int64_t m, int64_t n, std::complex<float> alpha, std::complex<float> const *a, int64_t lda, std::complex<float> const *x,
           int64_t incx, std::complex<float> beta, std::complex<float> *y, int64_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCgemv_64(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, (hipblasComplex *)&alpha, (hipblasComplex *)a, lda,
                               (hipblasComplex *)x, incx, (hipblasComplex *)&beta, (hipblasComplex *)y, incy));
#else
    hipblas_catch(hipblasCgemv_64(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, (hipComplex *)&alpha, (hipComplex *)a, lda,
                               (hipComplex *)x, incx, (hipComplex *)&beta, (hipComplex *)y, incy));
#endif

    gpu::stream_wait();
}

void zgemv(char transa, int64_t m, int64_t n, std::complex<double> alpha, std::complex<double> const *a, int64_t lda, std::complex<double> const *x,
           int64_t incx, std::complex<double> beta, std::complex<double> *y, int64_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZgemv_64(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, (hipblasDoubleComplex *)&alpha,
                               (hipblasDoubleComplex *)a, lda, (hipblasDoubleComplex *)x, incx, (hipblasDoubleComplex *)&beta,
                               (hipblasDoubleComplex *)y, incy));
#else
    hipblas_catch(hipblasZgemv_64(gpu::get_blas_handle(), vendor::char_to_op(transa), m, n, (hipDoubleComplex *)&alpha, (hipDoubleComplex *)a,
                               lda, (hipDoubleComplex *)x, incx, (hipDoubleComplex *)&beta, (hipDoubleComplex *)y, incy));
#endif

    gpu::stream_wait();
}

} // namespace einsums::blas::hip