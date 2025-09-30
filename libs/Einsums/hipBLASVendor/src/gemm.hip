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
    bool  notA = (std::tolower(transa) == 'n'), notB = (std::tolower(transb) == 'n');                                                      \
    int nrowa, nrowb;                                                                                                                    \
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
    if ((lda) < std::max((int)1, nrowa)) {                                                                                               \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The eighth argument (lda) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.",     \
            nrowa, lda);                                                                                                                   \
    }                                                                                                                                      \
    if ((ldb) < std::max((int)1, nrowb)) {                                                                                               \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The tenth argument (ldb) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.",      \
            nrowb, ldb);                                                                                                                   \
    }                                                                                                                                      \
    if ((ldc) < std::max((int)1, m)) {                                                                                                   \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The thirteenth argument (ldc) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.", \
            m, ldc);                                                                                                                       \
    }

void sgemm(char transa, char transb, int m, int n, int k, float alpha, float const *a, int lda, float const *b, int ldb,
           float beta, float *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    float *alpha_gpu = gpu::register_host_variable(alpha), *beta_gpu = gpu::register_host_variable(beta);

    hipblas_catch(hipblasSgemm(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k, alpha_gpu, a, lda,
                               b, ldb, beta_gpu, c, ldc));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, double const *a, int lda, double const *b, int ldb,
           double beta, double *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    double *alpha_gpu = gpu::register_host_variable(alpha), *beta_gpu = gpu::register_host_variable(beta);

    hipblas_catch(hipblasDgemm(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k, alpha_gpu, a, lda,
                               b, ldb, beta_gpu, c, ldc));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, std::complex<float> const *a, int lda,
           std::complex<float> const *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    hipblasComplex *alpha_gpu = (hipblasComplex *)gpu::register_host_variable(alpha),
                   *beta_gpu  = (hipblasComplex *)gpu::register_host_variable(beta);

    hipblas_catch(hipblasCgemm(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k, alpha_gpu,
                               (hipblasComplex *)a, lda, (hipblasComplex *)b, ldb, beta_gpu, (hipblasComplex *)c, ldc));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, std::complex<double> const *a, int lda,
           std::complex<double> const *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    hipblasDoubleComplex *alpha_gpu = (hipblasDoubleComplex *)gpu::register_host_variable(alpha),
                         *beta_gpu  = (hipblasDoubleComplex *)gpu::register_host_variable(beta);

    hipblas_catch(hipblasZgemm(gpu::get_blas_handle(), vendor::char_to_op(transa), vendor::char_to_op(transb), m, n, k, alpha_gpu,
                               (hipblasDoubleComplex *)a, lda, (hipblasDoubleComplex *)b, ldb, beta_gpu, (hipblasDoubleComplex *)c, ldc));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
    gpu::unregister_host_variable(beta);
}

} // namespace einsums::blas::hip