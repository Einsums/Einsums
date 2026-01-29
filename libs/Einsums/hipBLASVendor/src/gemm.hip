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

#define GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)                                                                                 \
    bool notA = (std::tolower(transa) == 'n'), notB = (std::tolower(transb) == 'n');                                                       \
    int64_t  nrowa, nrowb;                                                                                                                     \
                                                                                                                                           \
    if (notA) {                                                                                                                            \
        nrowa = m;                                                                                                                         \
    } else {                                                                                                                               \
        nrowa = k;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    if (notB) {                                                                                                                            \
        nrowb = k;                                                                                                                         \
    } else {                                                                                                                               \
        nrowb = n;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    if (!notA && std::tolower(transa) != 'c' && std::tolower(transa) != 't') {                                                             \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,                                                                                     \
                                "The first argument (transA) to gemm call was invalid! Expected n, t, or c, case-insensitive, got {}.",    \
                                transa);                                                                                                   \
    }                                                                                                                                      \
                                                                                                                                           \
    if (!notB && std::tolower(transb) != 'c' && std::tolower(transb) != 't') {                                                             \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,                                                                                     \
                                "The second argument (transB) to gemm call was invalid! Expected n, t, or c, case-insensitive, got {}.",   \
                                transb);                                                                                                   \
    }                                                                                                                                      \
                                                                                                                                           \
    if ((m) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The third argument (m) to gemm call was invalid! It must be greater than or equal to zero. Got {}.", m);  \
    }                                                                                                                                      \
    if ((n) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The fourth argument (n) to gemm call was invalid! It must be greater than or equal to zero. Got {}.", n); \
    }                                                                                                                                      \
    if ((k) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The fifth argument (k) to gemm call was invalid! It must be greater than or equal to zero. Got {}.", k);  \
    }                                                                                                                                      \
    if ((lda) < std::max((int64_t)1, nrowa)) {                                                                                                 \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The eighth argument (lda) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.",     \
            nrowa, lda);                                                                                                                   \
    }                                                                                                                                      \
    if ((ldb) < std::max((int64_t)1, nrowb)) {                                                                                                 \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The tenth argument (ldb) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.",      \
            nrowb, ldb);                                                                                                                   \
    }                                                                                                                                      \
    if ((ldc) < std::max((int64_t)1, m)) {                                                                                                     \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The thirteenth argument (ldc) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.", \
            m, ldc);                                                                                                                       \
    }

void sgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float const *a, int64_t lda, float const *b, int64_t ldb, float beta,
           float *c, int64_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    hipblas_catch(hipblasSgemm_64(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k, &alpha, a, lda, b,
                               ldb, &beta, c, ldc));

    gpu::stream_wait();
}

void dgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double const *a, int64_t lda, double const *b, int64_t ldb, double beta,
           double *c, int64_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    hipblas_catch(hipblasDgemm_64(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k, &alpha, a, lda, b,
                               ldb, &beta, c, ldc));

    gpu::stream_wait();
}

void cgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha, std::complex<float> const *a, int64_t lda,
           std::complex<float> const *b, int64_t ldb, std::complex<float> beta, std::complex<float> *c, int64_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCgemm_64(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k,
                               (hipblasComplex const *)&alpha, (hipblasComplex *)a, lda, (hipblasComplex *)b, ldb,
                               (hipblasComplex const *)&beta, (hipblasComplex *)c, ldc));
#else
    hipblas_catch(hipblasCgemm_64(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k,
                               (hipComplex const *)&alpha, (hipComplex *)a, lda, (hipComplex *)b, ldb, (hipComplex const *)&beta,
                               (hipComplex *)c, ldc));
#endif

    gpu::stream_wait();
}

void zgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha, std::complex<double> const *a, int64_t lda,
           std::complex<double> const *b, int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZgemm_64(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k,
                               (hipblasDoubleComplex const *)&alpha, (hipblasDoubleComplex *)a, lda, (hipblasDoubleComplex *)b, ldb,
                               (hipblasDoubleComplex const *)beta, (hipblasDoubleComplex *)c, ldc));
#else
    hipblas_catch(hipblasZgemm_64(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k,
                               (hipDoubleComplex const *)&alpha, (hipDoubleComplex *)a, lda, (hipDoubleComplex *)b, ldb,
                               (hipDoubleComplex const *)&beta, (hipDoubleComplex *)c, ldc));
#endif

    gpu::stream_wait();
}

} // namespace einsums::blas::hip