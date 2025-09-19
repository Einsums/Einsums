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

void saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y) {
    LabeledSection0();

    float *alpha_x_gpu;

    hip_catch(hipHostRegister((void *)&alpha_x, sizeof(float), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_x_gpu, (void *)&alpha_x, 0));

    hipblas_catch(hipblasSaxpy(gpu::get_blas_handle(), n, alpha_x_gpu, x, inc_x, y, inc_y));
    gpu::stream_wait();

    hip_catch(hipHostUnregister(&alpha_x));
}

void daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y) {
    LabeledSection0();

    double *alpha_x_gpu;

    hip_catch(hipHostRegister((void *)&alpha_x, sizeof(double), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_x_gpu, (void *)&alpha_x, 0));

    hipblas_catch(hipblasDaxpy(gpu::get_blas_handle(), n, alpha_x_gpu, x, inc_x, y, inc_y));
    gpu::stream_wait();

    hip_catch(hipHostUnregister(&alpha_x));
}

void caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    LabeledSection0();

    hipblasComplex *alpha_x_gpu;

    hip_catch(hipHostRegister((void *)&alpha_x, sizeof(hipComplex), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_x_gpu, (void *)&alpha_x, 0));

    hipblas_catch(hipblasCaxpy(gpu::get_blas_handle(), n, alpha_x_gpu, (hipblasComplex const *)x, inc_x, (hipblasComplex *)y, inc_y));
    gpu::stream_wait();

    hip_catch(hipHostUnregister(&alpha_x));
}

void zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    LabeledSection0();

    hipblasDoubleComplex *alpha_x_gpu;

    hip_catch(hipHostRegister((void *)&alpha_x, sizeof(hipDoubleComplex), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_x_gpu, (void *)&alpha_x, 0));

    hipblas_catch(
        hipblasZaxpy(gpu::get_blas_handle(), n, alpha_x_gpu, (hipblasDoubleComplex const *)x, inc_x, (hipblasDoubleComplex *)y, inc_y));
    gpu::stream_wait();

    hip_catch(hipHostUnregister(&alpha_x));
}

void saxpby(int_t const n, float const a, float const *x, int_t const incx, float const b, float *y, int_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        sscal(n, b, y, incy);
    }
    saxpy(n, a, x, incx, y, incy);
}

void daxpby(int_t const n, double const a, double const *x, int_t const incx, double const b, double *y, int_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        dscal(n, b, y, incy);
    }
    daxpy(n, a, x, incx, y, incy);
}

void caxpby(int_t const n, std::complex<float> const a, std::complex<float> const *x, int_t const incx, std::complex<float> const b,
            std::complex<float> *y, int_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        cscal(n, b, y, incy);
    }
    caxpy(n, a, x, incx, y, incy);
}

void zaxpby(int_t const n, std::complex<double> const a, std::complex<double> const *x, int_t const incx, std::complex<double> const b,
            std::complex<double> *y, int_t const incy) {
    LabeledSection0();
    if (incy == 0) {
        *y *= b;
    } else {
        zscal(n, b, y, incy);
    }
    zaxpy(n, a, x, incx, y, incy);
}

} // namespace einsums::blas::hip