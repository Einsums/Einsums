//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Options/Get.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::blas::int_t;

namespace {

template <typename T>
std::vector<T> random_values(std::size_t n, unsigned seed) {
    std::mt19937                     gen(seed);
    std::normal_distribution<double> dist;
    std::vector<T>                   x(n);
    for (auto &value : x) {
        if constexpr (einsums::IsComplexV<T>) {
            value = T(static_cast<einsums::RemoveComplexT<T>>(dist(gen)), static_cast<einsums::RemoveComplexT<T>>(dist(gen)));
        } else {
            value = static_cast<T>(dist(gen));
        }
    }
    return x;
}

/// Where BLAS finds element i of a vector of n elements: a negative increment walks it from the far end.
std::ptrdiff_t offset(int_t i, int_t n, int_t inc) {
    return inc >= 0 ? static_cast<std::ptrdiff_t>(i) * inc : static_cast<std::ptrdiff_t>(n - 1 - i) * -inc;
}

/// The number of elements a vector of n elements spans at this increment.
std::size_t extent(int_t n, int_t inc) {
    return n <= 0 ? 1 : 1 + static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(inc < 0 ? -inc : inc);
}

/// The definition, summed in long double, with no vendor routine involved. Also returns sum |x_i||y_i|,
/// which scales the rounding error any summation order may make.
template <typename T>
auto reference_dot(int_t n, T const *x, int_t incx, T const *y, int_t incy, bool conjugate) {
    using Wide = std::conditional_t<einsums::IsComplexV<T>, std::complex<long double>, long double>;
    Wide        sum{};
    long double scale = 0;
    for (int_t i = 0; i < n; i++) {
        Wide const xi = static_cast<Wide>(x[offset(i, n, incx)]);
        Wide const yi = static_cast<Wide>(y[offset(i, n, incy)]);
        if constexpr (einsums::IsComplexV<T>) {
            sum += (conjugate ? std::conj(xi) : xi) * yi;
        } else {
            sum += xi * yi;
        }
        scale += std::abs(xi) * std::abs(yi);
    }
    return std::pair{sum, scale};
}

template <typename T>
T vendor_dot(int_t n, T const *x, int_t incx, T const *y, int_t incy) {
    namespace vendor = einsums::blas::vendor;
    if constexpr (std::is_same_v<T, float>) {
        return vendor::sdot(n, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, double>) {
        return vendor::ddot(n, x, incx, y, incy);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return vendor::cdot(n, x, incx, y, incy);
    } else {
        return vendor::zdot(n, x, incx, y, incy);
    }
}

template <typename T>
T vendor_dotc(int_t n, T const *x, int_t incx, T const *y, int_t incy) {
    namespace vendor = einsums::blas::vendor;
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        return vendor::cdotc(n, x, incx, y, incy);
    } else {
        return vendor::zdotc(n, x, incx, y, incy);
    }
}

template <typename T, typename Wide>
void check_against_reference(T got, std::pair<Wide, long double> const &reference, int_t n) {
    // Any summation order is within n * eps of the scale; twice that leaves room for the vendor's
    // own rounding of each product. A wrong element or a missed conjugation is off by order one.
    long double const eps   = std::numeric_limits<einsums::RemoveComplexT<T>>::epsilon();
    long double const bound = 2 * static_cast<long double>(n) * eps * reference.second;
    CHECK(std::abs(static_cast<Wide>(got) - reference.first) <= bound);
}

/// Pin the element count at which the dots go parallel, and put the previous value back.
struct PinParallelThreshold {
    std::int64_t previous{einsums::config::get(einsums::option::HardwareOmpMinParallelElements)};
    explicit PinParallelThreshold(std::int64_t elements) {
        einsums::config::set(einsums::option::HardwareOmpMinParallelElements, elements);
    }
    PinParallelThreshold(PinParallelThreshold const &)            = delete;
    PinParallelThreshold &operator=(PinParallelThreshold const &) = delete;
    ~PinParallelThreshold() { einsums::config::set(einsums::option::HardwareOmpMinParallelElements, previous); }
};

} // namespace

// The dots are Einsums' own loops rather than calls into BLAS: a Fortran function returning a float
// or a complex value returns it differently from one BLAS build to the next (Accelerate's sdot
// returns a double, and f2c-style complex results come back through a hidden pointer), and a conda
// environment can swap the BLAS under a built library. So these check the BLAS definition itself:
// every increment sign, a zero increment, n = 0 and negative n, conjugation of x and not y, on both
// the serial and the threaded path.
TEMPLATE_TEST_CASE("vendor dots follow the BLAS definition", "[blas-vendor][dot]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T = TestType;

    // Zero sends every call down the threaded path and the largest value keeps every call serial.
    auto const threshold = GENERATE(std::int64_t{0}, (std::numeric_limits<std::int64_t>::max)());
    CAPTURE(threshold);
    PinParallelThreshold const pin{threshold};

    for (int_t const n : {0, 1, 2, 7, 64, 1000, 100003}) {
        for (auto const [incx, incy] : {std::pair<int_t, int_t>{1, 1}, {2, 3}, {-1, 1}, {1, -2}, {-3, -1}, {0, 1}, {1, 0}}) {
            CAPTURE(n, incx, incy);
            auto const x = random_values<T>(extent(n, incx), 11);
            auto const y = random_values<T>(extent(n, incy), 23);

            check_against_reference(vendor_dot(n, x.data(), incx, y.data(), incy), reference_dot(n, x.data(), incx, y.data(), incy, false),
                                    n);
            if constexpr (einsums::IsComplexV<T>) {
                check_against_reference(vendor_dotc(n, x.data(), incx, y.data(), incy),
                                        reference_dot(n, x.data(), incx, y.data(), incy, true), n);
            }
        }
    }

    // BLAS returns zero for a negative count rather than reading anything.
    auto const x = random_values<T>(1, 5);
    CHECK(vendor_dot(-3, x.data(), 1, x.data(), 1) == T{});
    if constexpr (einsums::IsComplexV<T>) {
        CHECK(vendor_dotc(-3, x.data(), 1, x.data(), 1) == T{});
    }
}

// cdotc conjugates its first argument. With x = i and y = 1 the two dots differ in sign, so a
// conjugation on the wrong side, or none, cannot pass.
TEMPLATE_TEST_CASE("vendor dotc conjugates x and not y", "[blas-vendor][dot]", std::complex<float>, std::complex<double>) {
    using T = TestType;
    std::array<T, 1> const x{T{0, 1}};
    std::array<T, 1> const y{T{1, 0}};
    CHECK(vendor_dot<T>(1, x.data(), 1, y.data(), 1) == T{0, 1});
    CHECK(vendor_dotc<T>(1, x.data(), 1, y.data(), 1) == T{0, -1});
    CHECK(vendor_dotc<T>(1, y.data(), 1, x.data(), 1) == T{0, 1});
}
