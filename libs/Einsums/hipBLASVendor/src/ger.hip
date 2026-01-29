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

#define ger_parameter_check(m, n, inc_x, inc_y, lda)                                                                                       \
    if ((m) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error, "First parameter (m) in ger call ({}) is less than zero.", m);                          \
    }                                                                                                                                      \
    if ((n) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error, "Second parameter (n) in ger call ({}) is less than zero.", n);                         \
    }                                                                                                                                      \
    if ((inc_x) == 0) {                                                                                                                    \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Fifth parameter (inc_x) in ger call ({}) is zero.", inc_x);                        \
    }                                                                                                                                      \
    if ((inc_y) == 0) {                                                                                                                    \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Seventh parameter (inc_y) in ger call ({}) is zero.", inc_y);                      \
    }                                                                                                                                      \
    if ((lda) < std::max(int64_t{1}, m)) {                                                                                                     \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Ninth parameter (lda) in ger call ({}) is less than max(1, n ({})).", lda, n);        \
    }

void sger(int64_t m, int64_t n, float alpha, float const *x, int64_t inc_x, float const *y, int64_t inc_y, float *a, int64_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    hipblas_catch(hipblasSger_64(gpu::get_blas_handle(), m, n, &alpha, x, inc_x, y, inc_y, a, lda));

    gpu::stream_wait();
}

void dger(int64_t m, int64_t n, double alpha, double const *x, int64_t inc_x, double const *y, int64_t inc_y, double *a, int64_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    hipblas_catch(hipblasDger_64(gpu::get_blas_handle(), m, n, &alpha, x, inc_x, y, inc_y, a, lda));

    gpu::stream_wait();
}

void cger(int64_t m, int64_t n, std::complex<float> alpha, std::complex<float> const *x, int64_t inc_x, std::complex<float> const *y, int64_t inc_y,
          std::complex<float> *a, int64_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCgeru_64(gpu::get_blas_handle(), m, n, (hipblasComplex const *)&alpha, (hipblasComplex *)x, inc_x,
                               (hipblasComplex *)y, inc_y, (hipblasComplex *)a, lda));
#else
    hipblas_catch(hipblasCgeru_64(gpu::get_blas_handle(), m, n, (hipComplex const *)&alpha, (hipComplex *)x, inc_x, (hipComplex *)y, inc_y,
                               (hipComplex *)a, lda));
#endif

    gpu::stream_wait();
}

void zger(int64_t m, int64_t n, std::complex<double> alpha, std::complex<double> const *x, int64_t inc_x, std::complex<double> const *y, int64_t inc_y,
          std::complex<double> *a, int64_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZgeru_64(gpu::get_blas_handle(), m, n, (hipblasDoubleComplex const *)&alpha, (hipblasDoubleComplex *)x, inc_x,
                               (hipblasDoubleComplex *)y, inc_y, (hipblasDoubleComplex *)a, lda));
#else
    hipblas_catch(hipblasZgeru_64(gpu::get_blas_handle(), m, n, (hipDoubleComplex const *)&alpha, (hipDoubleComplex *)x, inc_x,
                               (hipDoubleComplex *)y, inc_y, (hipDoubleComplex *)a, lda));
#endif

    gpu::stream_wait();
}

void cgerc(int64_t m, int64_t n, std::complex<float> alpha, std::complex<float> const *x, int64_t inc_x, std::complex<float> const *y, int64_t inc_y,
           std::complex<float> *a, int64_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCgerc_64(gpu::get_blas_handle(), m, n, (hipblasComplex const *)&alpha, (hipblasComplex *)x, inc_x,
                               (hipblasComplex *)y, inc_y, (hipblasComplex *)a, lda));
#else
    hipblas_catch(hipblasCgerc_64(gpu::get_blas_handle(), m, n, (hipComplex const *)&alpha, (hipComplex *)x, inc_x, (hipComplex *)y, inc_y,
                               (hipComplex *)a, lda));
#endif

    gpu::stream_wait();
}

void zgerc(int64_t m, int64_t n, std::complex<double> alpha, std::complex<double> const *x, int64_t inc_x, std::complex<double> const *y, int64_t inc_y,
           std::complex<double> *a, int64_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZgerc_64(gpu::get_blas_handle(), m, n, (hipblasDoubleComplex const *)&alpha, (hipblasDoubleComplex *)x, inc_x,
                               (hipblasDoubleComplex *)y, inc_y, (hipblasDoubleComplex *)a, lda));
#else
    hipblas_catch(hipblasZgerc_64(gpu::get_blas_handle(), m, n, (hipDoubleComplex const *)&alpha, (hipDoubleComplex *)x, inc_x,
                               (hipDoubleComplex *)y, inc_y, (hipDoubleComplex *)a, lda));
#endif

    gpu::stream_wait();
}

} // namespace einsums::blas::hip