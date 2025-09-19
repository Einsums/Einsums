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
    if ((lda) < std::max(int{1}, m)) {                                                                                                   \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Ninth parameter (lda) in ger call ({}) is less than max(1, n ({})).", lda, n);        \
    }

void sger(int m, int n, float alpha, float const *x, int inc_x, float const *y, int inc_y, float *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    float *alpha_gpu = gpu::register_host_variable(alpha);

    hipblas_catch(hipblasSger(gpu::get_blas_handle(), m, n, alpha_gpu, x, inc_x, y, inc_y, a, lda));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
}

void dger(int m, int n, double alpha, double const *x, int inc_x, double const *y, int inc_y, double *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    double *alpha_gpu = gpu::register_host_variable(alpha);

    hipblas_catch(hipblasDger(gpu::get_blas_handle(), m, n, alpha_gpu, x, inc_x, y, inc_y, a, lda));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
}

void cger(int m, int n, std::complex<float> alpha, std::complex<float> const *x, int inc_x, std::complex<float> const *y, int inc_y,
          std::complex<float> *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    hipblasComplex *alpha_gpu = (hipblasComplex *)gpu::register_host_variable(alpha);

    hipblas_catch(hipblasCgeru(gpu::get_blas_handle(), m, n, alpha_gpu, (hipblasComplex *)x, inc_x, (hipblasComplex *)y, inc_y,
                               (hipblasComplex *)a, lda));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
}

void zger(int m, int n, std::complex<double> alpha, std::complex<double> const *x, int inc_x, std::complex<double> const *y,
          int inc_y, std::complex<double> *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    hipblasDoubleComplex *alpha_gpu = (hipblasDoubleComplex *)gpu::register_host_variable(alpha);

    hipblas_catch(hipblasZgeru(gpu::get_blas_handle(), m, n, alpha_gpu, (hipblasDoubleComplex *)x, inc_x, (hipblasDoubleComplex *)y, inc_y,
                               (hipblasDoubleComplex *)a, lda));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
}

void cgerc(int m, int n, std::complex<float> alpha, std::complex<float> const *x, int inc_x, std::complex<float> const *y,
           int inc_y, std::complex<float> *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    hipblasComplex *alpha_gpu = (hipblasComplex *)gpu::register_host_variable(alpha);

    hipblas_catch(hipblasCgerc(gpu::get_blas_handle(), m, n, alpha_gpu, (hipblasComplex *)x, inc_x, (hipblasComplex *)y, inc_y,
                               (hipblasComplex *)a, lda));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
}

void zgerc(int m, int n, std::complex<double> alpha, std::complex<double> const *x, int inc_x, std::complex<double> const *y,
           int inc_y, std::complex<double> *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    hipblasDoubleComplex *alpha_gpu = (hipblasDoubleComplex *)gpu::register_host_variable(alpha);

    hipblas_catch(hipblasZgerc(gpu::get_blas_handle(), m, n, alpha_gpu, (hipblasDoubleComplex *)x, inc_x, (hipblasDoubleComplex *)y, inc_y,
                               (hipblasDoubleComplex *)a, lda));

    gpu::stream_wait();

    gpu::unregister_host_variable(alpha);
}

} // namespace einsums::blas::hip