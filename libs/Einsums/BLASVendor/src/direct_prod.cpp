//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/LabeledSection.hpp>

namespace einsums::blas::vendor {

void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::sdirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::sdirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::ddirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::ddirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy,
              std::complex<float> *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::cdirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::cdirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy,
              std::complex<double> *z, int_t incz) {
    LabeledSection0();

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks    = n / 64;
        auto remaining = n % 64;
        auto offset    = 64 * blocks;

        if (blocks != 0) {
            EINSUMS_OMP_PARALLEL_FOR
            for (int_t i = 0; i < blocks; i++) {
                ::zdirprod_kernel(64, alpha, x + i * 64, y + i * 64, z + i * 64);
            }
        }

        if (remaining != 0) {
            ::zdirprod_kernel(remaining, alpha, x + offset, y + offset, z + offset);
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int_t i = 0; i < n; i++) {
            z[i * incz] += alpha * x[i * incx] * y[i * incy];
        }
    }
}

} // namespace einsums::blas::vendor