//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams.hpp>

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hipsolver/internal/hipsolver-types.h>

namespace einsums {

namespace blas {
namespace gpu {

template <typename T>
int iamax(hipblasHandle_t handle, int n, T const *x, int incx);

template <>
inline int iamax(hipblasHandle_t handle, int n, float const *x, int incx) {
    int result;

    hipblas_catch(hipblasIsamax(handle, n, x, incx, &result));

    return result;
}

template <>
inline int iamax(hipblasHandle_t handle, int n, double const *x, int incx) {
    int result;

    hipblas_catch(hipblasIdamax(handle, n, x, incx, &result));

    return result;
}

template <>
inline int iamax(hipblasHandle_t handle, int n, std::complex<float> const *x, int incx) {
    int result;

    hipblas_catch(hipblasIcamax(handle, n, (hipblasComplex *)x, incx, &result));

    return result;
}

template <>
inline int iamax(hipblasHandle_t handle, int n, std::complex<double> const *x, int incx) {
    int result;

    hipblas_catch(hipblasIzamax(handle, n, (hipblasDoubleComplex *)x, incx, &result));

    return result;
}

template <typename T>
inline int iamin(hipblasHandle_t handle, int n, T const *x, int incx);

template <>
inline int iamin(hipblasHandle_t handle, int n, float const *x, int incx) {
    int result;

    hipblas_catch(hipblasIsamin(handle, n, x, incx, &result));

    return result;
}

template <>
inline int iamin(hipblasHandle_t handle, int n, double const *x, int incx) {
    int result;

    hipblas_catch(hipblasIdamin(handle, n, x, incx, &result));

    return result;
}

template <>
inline int iamin(hipblasHandle_t handle, int n, std::complex<float> const *x, int incx) {
    int result;

    hipblas_catch(hipblasIcamin(handle, n, (hipblasComplex *)x, incx, &result));

    return result;
}

template <>
inline int iamin(hipblasHandle_t handle, int n, std::complex<double> const *x, int incx) {
    int result;

    hipblas_catch(hipblasIzamin(handle, n, (hipblasDoubleComplex *)x, incx, &result));

    return result;
}

template <typename T>
void axpy(hipblasHandle_t handle, int n, T const *alpha, T const *x, int incx, T *y, int incy);

template <>
inline void axpy(hipblasHandle_t handle, int n, float const *alpha, float const *x, int incx, float *y, int incy) {
    hipblas_catch(hipblasSaxpy(handle, n, alpha, x, incx, y, incy));
}

template <>
inline void axpy(hipblasHandle_t handle, int n, double const *alpha, double const *x, int incx, double *y, int incy) {
    hipblas_catch(hipblasDaxpy(handle, n, alpha, x, incx, y, incy));
}

template <>
inline void axpy(hipblasHandle_t handle, int n, std::complex<float> const *alpha, std::complex<float> const *x, int incx,
                 std::complex<float> *y, int incy) {
    hipblas_catch(hipblasCaxpy(handle, n, (hipblasComplex *)alpha, (hipblasComplex *)x, incx, (hipblasComplex *)y, incy));
}

template <>
inline void axpy(hipblasHandle_t handle, int n, std::complex<double> const *alpha, std::complex<double> const *x, int incx,
                 std::complex<double> *y, int incy) {
    hipblas_catch(hipblasZaxpy(handle, n, (hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)x, incx, (hipblasDoubleComplex *)y, incy));
}

template <typename T>
void scal(hipblasHandle_t handle, int n, T const *alpha, T *x, int incx);

template <>
inline void scal(hipblasHandle_t handle, int n, float const *alpha, float *x, int incx) {
    hipblas_catch(hipblasSscal(handle, n, alpha, x, incx));
}

template <>
inline void scal(hipblasHandle_t handle, int n, double const *alpha, double *x, int incx) {
    hipblas_catch(hipblasDscal(handle, n, alpha, x, incx));
}

template <>
inline void scal(hipblasHandle_t handle, int n, std::complex<float> const *alpha, std::complex<float> *x, int incx) {
    hipblas_catch(hipblasCscal(handle, n, (hipblasComplex *)alpha, (hipblasComplex *)x, incx));
}

template <>
inline void scal(hipblasHandle_t handle, int n, std::complex<double> const *alpha, std::complex<double> *x, int incx) {
    hipblas_catch(hipblasZscal(handle, n, (hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)x, incx));
}

} // namespace gpu
} // namespace blas

} // namespace einsums