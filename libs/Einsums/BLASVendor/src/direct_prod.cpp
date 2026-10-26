//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

namespace einsums::blas::vendor {

void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    EINSUMS_OMP_SIMD
    for (int_t i = 0; i < n; i++) {
        z[i * incz] += alpha * x[i * incx] * y[i * incy];
    }
}

void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    EINSUMS_OMP_SIMD
    for (int_t i = 0; i < n; i++) {
        z[i * incz] += alpha * x[i * incx] * y[i * incy];
    }
}

void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy,
              std::complex<float> *z, int_t incz) {
    EINSUMS_OMP_SIMD
    for (int_t i = 0; i < n; i++) {
        z[i * incz] += alpha * x[i * incx] * y[i * incy];
    }
}

void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy,
              std::complex<double> *z, int_t incz) {
    EINSUMS_OMP_SIMD
    for (int_t i = 0; i < n; i++) {
        z[i * incz] += alpha * x[i * incx] * y[i * incy];
    }
}

} // namespace einsums::blas::vendor