//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <complex>
#include <cstddef>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

// The dots are loops here rather than calls into BLAS. A Fortran function that returns a float or a
// complex value returns it differently from one BLAS build to the next: Accelerate's sdot returns a
// double, and f2c-style builds hand complex results back through a hidden first argument. Which
// convention applies cannot be settled when Einsums is built, because a conda environment swaps the
// BLAS under a built library. A dot is bound by memory bandwidth, so a loop the compiler vectorizes
// gives up nothing to a vendor kernel.

namespace {

// Where element 0 of a vector of n elements sits: BLAS walks a negative increment from the far end.
std::ptrdiff_t first_element(int_t n, int_t inc) {
    return inc >= 0 ? 0 : static_cast<std::ptrdiff_t>(n - 1) * -inc;
}

template <typename T>
T real_dot(int_t n, T const *x, int_t incx, T const *y, int_t incy) {
    // The n guard keeps an empty dot from reading anything; an empty dot is zero.
    if (n <= 0)
        return T{0};

    T const *const       x0 = x + first_element(n, incx);
    T const *const       y0 = y + first_element(n, incy);
    std::ptrdiff_t const sx = incx;
    std::ptrdiff_t const sy = incy;

    // `parallel:` keeps the condition off the simd half, which a bare `if` would also switch off.
    T sum{0};
    EINSUMS_OMP_PRAGMA(parallel for simd reduction(+ : sum)
                           if (parallel : static_cast<std::size_t>(n) >= hardware::omp_min_parallel_elements() && !omp_in_parallel()))
    for (int_t i = 0; i < n; i++) {
        sum += x0[i * sx] * y0[i * sy];
    }
    return sum;
}

// The real and imaginary parts are summed as separate reals, which every compiler reduces and
// vectorizes, and std::complex is laid out as the pair of them.
template <bool Conjugate, typename T>
std::complex<T> complex_dot(int_t n, std::complex<T> const *x, int_t incx, std::complex<T> const *y, int_t incy) {
    if (n <= 0)
        return std::complex<T>{};

    T const *const       x0 = reinterpret_cast<T const *>(x + first_element(n, incx));
    T const *const       y0 = reinterpret_cast<T const *>(y + first_element(n, incy));
    std::ptrdiff_t const sx = 2 * static_cast<std::ptrdiff_t>(incx);
    std::ptrdiff_t const sy = 2 * static_cast<std::ptrdiff_t>(incy);

    T re{0};
    T im{0};
    EINSUMS_OMP_PRAGMA(parallel for simd reduction(+ : re, im)
                           if (parallel : static_cast<std::size_t>(n) >= hardware::omp_min_parallel_elements() && !omp_in_parallel()))
    for (int_t i = 0; i < n; i++) {
        T const a = x0[i * sx];
        T const b = x0[i * sx + 1];
        T const c = y0[i * sy];
        T const d = y0[i * sy + 1];
        // (a + bi)(c + di), with b negated when x is conjugated.
        if constexpr (Conjugate) {
            re += a * c + b * d;
            im += a * d - b * c;
        } else {
            re += a * c - b * d;
            im += a * d + b * c;
        }
    }
    return {re, im};
}

} // namespace

auto sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    WAGGLE_ZONE_FUNC();
    return real_dot(n, x, incx, y, incy);
}

auto ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    WAGGLE_ZONE_FUNC();
    return real_dot(n, x, incx, y, incy);
}

auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    WAGGLE_ZONE_FUNC();
    return complex_dot<false>(n, x, incx, y, incy);
}

auto zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    WAGGLE_ZONE_FUNC();
    return complex_dot<false>(n, x, incx, y, incy);
}

auto cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    WAGGLE_ZONE_FUNC();
    return complex_dot<true>(n, x, incx, y, incy);
}

auto zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    WAGGLE_ZONE_FUNC();
    return complex_dot<true>(n, x, incx, y, incy);
}

EINSUMS_NAMESPACE_END(blas::vendor)
