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

auto sdot(int64_t n, float const *x, int64_t incx, float const *y, int64_t incy) -> float {
    LabeledSection0();

    float result;

    hipblas_catch(hipblasSdot_64(gpu::get_blas_handle(), n, x, incx, y, incy, &result));

    gpu::stream_wait();

    return result;
}

auto ddot(int64_t n, double const *x, int64_t incx, double const *y, int64_t incy) -> double {
    LabeledSection0();

    double result;

    hipblas_catch(hipblasDdot_64(gpu::get_blas_handle(), n, x, incx, y, incy, &result));

    gpu::stream_wait();

    return result;
}

// We implement the cdotu as the default for cdot.
auto cdot(int64_t n, std::complex<float> const *x, int64_t incx, std::complex<float> const *y, int64_t incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCdotu_64(gpu::get_blas_handle(), n, (hipblasComplex const *)x, incx, (hipblasComplex const *)y, incy,
                               (hipblasComplex *)&result));
#else
    hipblas_catch(hipblasCdotu_64(gpu::get_blas_handle(), n, (hipComplex const *)x, incx, (hipComplex const *)y, incy, (hipComplex *)&result));
#endif

    gpu::stream_wait();

    return result;
}

// We implement the zdotu as the default for cdot.
auto zdot(int64_t n, std::complex<double> const *x, int64_t incx, std::complex<double> const *y, int64_t incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZdotu_64(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, incx, (hipblasDoubleComplex const *)y, incy,
                               (hipblasDoubleComplex *)&result));
#else
    hipblas_catch(hipblasZdotu_64(gpu::get_blas_handle(), n, (hipDoubleComplex const *)x, incx, (hipDoubleComplex const *)y, incy,
                               (hipDoubleComplex *)&result));
#endif

    gpu::stream_wait();

    return result;
}

auto cdotc(int64_t n, std::complex<float> const *x, int64_t incx, std::complex<float> const *y, int64_t incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCdotc_64(gpu::get_blas_handle(), n, (hipblasComplex const *)x, incx, (hipblasComplex const *)y, incy,
                               (hipblasComplex *)&result));
#else
    hipblas_catch(hipblasCdotc_64(gpu::get_blas_handle(), n, (hipComplex const *)x, incx, (hipComplex const *)y, incy, (hipComplex *)&result));
#endif

    gpu::stream_wait();

    return result;
}

auto zdotc(int64_t n, std::complex<double> const *x, int64_t incx, std::complex<double> const *y, int64_t incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZdotc_64(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, incx, (hipblasDoubleComplex const *)y, incy,
                               (hipblasDoubleComplex *)&result));
#else
    hipblas_catch(hipblasZdotc_64(gpu::get_blas_handle(), n, (hipDoubleComplex const *)x, incx, (hipDoubleComplex const *)y, incy,
                               (hipDoubleComplex *)&result));
#endif

    gpu::stream_wait();

    return result;
}

} // namespace einsums::blas::hip