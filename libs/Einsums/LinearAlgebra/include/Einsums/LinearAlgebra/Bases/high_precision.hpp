//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <complex>
namespace einsums {
namespace linear_algebra {
namespace detail {

constexpr float mult_error(float x, float y) {
    return std::fma(x, y, -x * y);
}

constexpr double mult_error(double x, double y) {
    return std::fma(x, y, -x * y);
}

constexpr float fma(float const &x, float const &y, float const &sum) {
    return std::fma(x, y, sum);
}

constexpr double fma(double const &x, double const &y, double const &sum) {
    return std::fma(x, y, sum);
}

constexpr std::complex<float> fma(std::complex<float> const &x, std::complex<float> const &y, std::complex<float> const &sum) {
    float const real_real = std::real(x) * std::real(y);
    float const imag_imag = std::imag(x) * std::imag(y);
    float const real_imag = std::real(x) * std::imag(y);
    float const imag_real = std::imag(x) * std::imag(y);

    float const real = real_real - imag_imag;
    float const imag = real_imag + imag_real;

    float const real_real_error = mult_error(std::real(x), std::real(y));
    float const imag_imag_error = mult_error(std::imag(x), std::imag(y));
    float const real_imag_error = mult_error(std::real(x), std::imag(y));
    float const imag_real_error = mult_error(std::imag(x), std::real(y));

    float const real_error = real_real_error - imag_imag_error;
    float const imag_error = real_imag_error + imag_real_error;

    return std::complex<float>{real + real_error, imag + imag_error};
}

constexpr std::complex<double> fma(std::complex<double> const &x, std::complex<double> const &y, std::complex<double> const &sum) {
    double const real_real = std::real(x) * std::real(y);
    double const imag_imag = std::imag(x) * std::imag(y);
    double const real_imag = std::real(x) * std::imag(y);
    double const imag_real = std::imag(x) * std::imag(y);

    double const real = real_real - imag_imag;
    double const imag = real_imag + imag_real;

    double const real_real_error = mult_error(std::real(x), std::real(y));
    double const imag_imag_error = mult_error(std::imag(x), std::imag(y));
    double const real_imag_error = mult_error(std::real(x), std::imag(y));
    double const imag_real_error = mult_error(std::imag(x), std::real(y));

    double const real_error = real_real_error - imag_imag_error;
    double const imag_error = real_imag_error + imag_real_error;

    return std::complex<double>{real + real_error, imag + imag_error};
}

constexpr std::complex<float> fma(std::complex<float> const &x, float const &y, std::complex<float> const &sum) {
    return std::complex<float>{std::fma(std::real(x), y, std::real(sum)), std::fma(std::imag(x), y, std::imag(sum))};
}

constexpr std::complex<double> fma(std::complex<double> const &x, double const &y, std::complex<double> const &sum) {
    return std::complex<double>{std::fma(std::real(x), y, std::real(sum)), std::fma(std::imag(x), y, std::imag(sum))};
}

constexpr std::complex<float> fma(float const &x, std::complex<float> const &y, std::complex<float> const &sum) {
    return detail::fma(y, x, sum);
}

constexpr std::complex<double> fma(double const &x, std::complex<double> const &y, std::complex<double> const &sum) {
    return detail::fma(y, x, sum);
}

template <typename T>
constexpr T triple_product(T const &x, T const &y, T const &z) {
    T const first_prod  = x * y;
    T const first_error = detail::fma(x, y, -first_prod);

    T const second_prod  = z * first_prod;
    T const second_error = z * first_error;
    T const third_error  = detail::fma(z, first_prod, -second_prod);

    T const error = second_error + third_error;

    return second_prod + error;
}

template <typename T>
inline void add_scale(T value, T &big_sum, T &medium_sum, T &small_sum, bool &not_big, bool ignore = false) {
    constexpr T sfmin  = std::numeric_limits<T>::min();
    constexpr T small  = 1 / std::numeric_limits<T>::max();
    constexpr T smlnum = (small > sfmin) ? small * (1 + std::numeric_limits<T>::epsilon()) : sfmin;
    constexpr T bignum = 1 / smlnum;

    auto const ax = std::abs(value);

    if (ax > bignum) {
        big_sum += value * smlnum;
        not_big = false;
    } else if (ax < smlnum) {
        if (not_big) {
            small_sum += value * bignum;
        }
    } else {
        medium_sum += value;
    }
}

template <typename T>
inline void add_scale(std::complex<T> value, std::complex<T> &big_sum, std::complex<T> &medium_sum, std::complex<T> &small_sum,
                      bool &not_big_re, bool &not_big_im) {
    // This is allowed and guaranteed by the C++ standard.
    auto big_array    = reinterpret_cast<T(&)[2]>(big_sum);
    auto medium_array = reinterpret_cast<T(&)[2]>(medium_sum);
    auto small_array  = reinterpret_cast<T(&)[2]>(small_sum);
    add_scale(value.real(), big_array[0], medium_array[0], small_array[0], not_big_re);
    add_scale(value.imag(), big_array[1], medium_array[1], small_array[1], not_big_im);
}

template <typename T>
inline T combine_accum(T big_sum, T medium_sum, T small_sum) {
#pragma float_control(precise, on)
    constexpr T sfmin  = std::numeric_limits<T>::min();
    constexpr T small  = 1 / std::numeric_limits<T>::max();
    constexpr T smlnum = (small > sfmin) ? small * (1 + std::numeric_limits<T>::epsilon()) : sfmin;
    constexpr T bignum = 1 / smlnum;

    if (big_sum > T{0.0}) {
        if (std::abs(medium_sum) > T{0.0} || std::isnan(medium_sum)) {
            big_sum += medium_sum * smlnum;
        }
        return big_sum / smlnum;
    } else if (std::abs(small_sum) > T{0.0}) {
        if (std::abs(medium_sum) > T{0.0} || std::isnan(medium_sum)) {
            medium_sum += small_sum * bignum;
            return medium_sum;
        } else {
            return small_sum * bignum;
        }
    } else {
        return medium_sum;
    }
}

template <typename T>
inline std::complex<T> combine_accum(std::complex<T> big_sum, std::complex<T> medium_sum, std::complex<T> small_sum) {
    return std::complex<T>{combine_accum(big_sum.real(), medium_sum.real(), small_sum.real()),
                           combine_accum(big_sum.imag(), medium_sum.imag(), small_sum.imag())};
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums