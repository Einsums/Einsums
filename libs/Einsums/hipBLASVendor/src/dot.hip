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

auto sdot(int n, float const *x, int incx, float const *y, int incy) -> float {
    LabeledSection0();

    float result;

    hipblas_catch(hipblasSdot(gpu::get_blas_handle(), n, x, incx, y, incy, &result));

    gpu::stream_wait();

    return result;
}

auto ddot(int n, double const *x, int incx, double const *y, int incy) -> double {
    LabeledSection0();

    double result;

    hipblas_catch(hipblasDdot(gpu::get_blas_handle(), n, x, incx, y, incy, &result));

    gpu::stream_wait();

    return result;
}

// We implement the cdotu as the default for cdot.
auto cdot(int n, std::complex<float> const *x, int incx, std::complex<float> const *y, int incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;

    hipblas_catch(hipblasCdotu(gpu::get_blas_handle(), n, (hipblasComplex const *)x, incx, (hipblasComplex const *)y, incy,
                               (hipblasComplex *)&result));

    gpu::stream_wait();

    return result;
}

// We implement the zdotu as the default for cdot.
auto zdot(int n, std::complex<double> const *x, int incx, std::complex<double> const *y, int incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;

    hipblas_catch(hipblasZdotu(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, incx, (hipblasDoubleComplex const *)y, incy,
                               (hipblasDoubleComplex *)&result));

    gpu::stream_wait();

    return result;
}

auto cdotc(int n, std::complex<float> const *x, int incx, std::complex<float> const *y, int incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;

    hipblas_catch(hipblasCdotc(gpu::get_blas_handle(), n, (hipblasComplex const *)x, incx, (hipblasComplex const *)y, incy,
                               (hipblasComplex *)&result));

    gpu::stream_wait();

    return result;
}

auto zdotc(int n, std::complex<double> const *x, int incx, std::complex<double> const *y, int incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;

    hipblas_catch(hipblasZdotc(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, incx, (hipblasDoubleComplex const *)y, incy,
                               (hipblasDoubleComplex *)&result));

    gpu::stream_wait();

    return result;
}

} // namespace einsums::blas::hip