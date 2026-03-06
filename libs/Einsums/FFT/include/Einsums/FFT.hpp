//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/Tensor.hpp>

namespace einsums::fft {
#ifndef DOXYGEN

namespace detail {

/*************************************
 * Real or complex -> complex        *
 *************************************/
void EINSUMS_EXPORT scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result);

void EINSUMS_EXPORT dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result);
void EINSUMS_EXPORT zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result);

/*************************************
 * Real or complex <- complex        *
 *************************************/
void EINSUMS_EXPORT csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result);
void EINSUMS_EXPORT zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result);

void EINSUMS_EXPORT ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result);

} // namespace detail
#endif

/**
 * @brief Compute the frequencies for a given number of elements.
 *
 * @param n The number of elements.
 * @param d The scale factor for the frequencies. The highest frequency will be @f$\frac{1}{2d}@f$.
 *
 * @return A tensor containing the positive and negative frequencies for each value position in the result of
 * a Fourier transform.
 *
 * @versionadded{1.0.0}
 */
auto EINSUMS_EXPORT fftfreq(int n, double d = 1.0) -> Tensor<double, 1>;

/**
 * @brief Perform the fast Fourier transform on a vector.
 *
 * @param a The input vector.
 * @param result The output vector.
 *
 * @versionadded{1.0.0}
 */
inline void fft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::scfft(a, result);
}

/// @copydoc fft(Tensor<float,1> const &,Tensor<std::complex<float>,1> *)
inline void fft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccfft(a, result);
}

/// @copydoc fft(Tensor<float,1> const &,Tensor<std::complex<float>,1> *)
inline void fft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::dzfft(a, result);
}

/// @copydoc fft(Tensor<float,1> const &,Tensor<std::complex<float>,1> *)
inline void fft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzfft(a, result);
}

/**
 * @brief Perform the inverse fast Fourier transform on a vector.
 *
 * @param a The input vector.
 * @param result The output vector.
 *
 * @versionadded{1.0.0}
 */
inline void ifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    detail::csifft(a, result);
}

/// @copydoc ifft(Tensor<std::complex<float>,1> const &,Tensor<float,1> *)
inline void ifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    detail::zdifft(a, result);
}

/// @copydoc ifft(Tensor<std::complex<float>,1> const &,Tensor<float,1> *)
inline void ifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccifft(a, result);
}

/// @copydoc ifft(Tensor<std::complex<float>,1> const &,Tensor<float,1> *)
inline void ifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzifft(a, result);
}

} // namespace einsums::fft