//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/GPUStreams.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include "Common.hpp"

namespace einsums::blas::hip {

void saxpy(int64_t n, float alpha, float const *x, int64_t inc_x, float *y, int64_t inc_y) {
    LabeledSection0();

    hipblas_catch(hipblasSaxpy_64(gpu::get_blas_handle(), n, &alpha, x, inc_x, y, inc_y));
    gpu::stream_wait();
}

void daxpy(int64_t n, double alpha, double const *x, int64_t inc_x, double *y, int64_t inc_y) {
    LabeledSection0();

    hipblas_catch(hipblasDaxpy_64(gpu::get_blas_handle(), n, &alpha, x, inc_x, y, inc_y));
    gpu::stream_wait();
}

void caxpy(int64_t n, std::complex<float> alpha, std::complex<float> const *x, int64_t inc_x, std::complex<float> *y, int64_t inc_y) {
    LabeledSection0();

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCaxpy_64(gpu::get_blas_handle(), n, (hipblasComplex const *)&alpha, (hipblasComplex const *)x, inc_x,
                               (hipblasComplex *)y, inc_y));
#else
    hipblas_catch(
        hipblasCaxpy_64(gpu::get_blas_handle(), n, (hipComplex const *)&alpha, (hipComplex const *)x, inc_x, (hipComplex *)y, inc_y));
#endif
    gpu::stream_wait();
}

void zaxpy(int64_t n, std::complex<double> alpha, std::complex<double> const *x, int64_t inc_x, std::complex<double> *y, int64_t inc_y) {
    LabeledSection0();

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZaxpy_64(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)&alpha, (hipblasDoubleComplex const *)x, inc_x,
                               (hipblasDoubleComplex *)y, inc_y));
#else
    hipblas_catch(hipblasZaxpy_64(gpu::get_blas_handle(), n, (hipDoubleComplex const *)&alpha, (hipDoubleComplex const *)x, inc_x,
                               (hipDoubleComplex *)y, inc_y));
#endif
    gpu::stream_wait();
}

void saxpby(int64_t const n, float const a, float const *x, int64_t const incx, float const b, float *y, int64_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        sscal(n, b, y, incy);
    }
    saxpy(n, a, x, incx, y, incy);
}

void daxpby(int64_t const n, double const a, double const *x, int64_t const incx, double const b, double *y, int64_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        dscal(n, b, y, incy);
    }
    daxpy(n, a, x, incx, y, incy);
}

void caxpby(int64_t const n, std::complex<float> const a, std::complex<float> const *x, int64_t const incx, std::complex<float> const b,
            std::complex<float> *y, int64_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        cscal(n, b, y, incy);
    }
    caxpy(n, a, x, incx, y, incy);
}

void zaxpby(int64_t const n, std::complex<double> const a, std::complex<double> const *x, int64_t const incx, std::complex<double> const b,
            std::complex<double> *y, int64_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        zscal(n, b, y, incy);
    }
    zaxpy(n, a, x, incx, y, incy);
}

} // namespace einsums::blas::hip