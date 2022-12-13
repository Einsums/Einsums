#pragma once

#include "einsums/Tensor.hpp"
#include "einsums/_Export.hpp"

// #include <mkl_dfti.h>

namespace einsums::fft {

namespace detail {}

auto EINSUMS_EXPORT fftfreq(int n, double d = 1.0) -> Tensor<double, 1>;

namespace detail {
void EINSUMS_EXPORT scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result);

void EINSUMS_EXPORT dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result);
void EINSUMS_EXPORT zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result);
} // namespace detail

inline void fft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result) {
    detail::scfft(a, result);
}

inline void fft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccfft(a, result);
}

inline void fft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result) {
    detail::dzfft(a, result);
}

inline void fft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzfft(a, result);
}

/**
 * Compute the one-dimensional discrete Fourier Transform.
 *
 * This function computes the one-dimensional *n*-point discrete Fourier
 * Transform (DFT) with the efficient Fast Fourier Transform (FFT)
 * algorithm.
 *
 * \param a Input array, can be complex.
 *
 * \returns complex tensor
 */

} // namespace einsums::fft