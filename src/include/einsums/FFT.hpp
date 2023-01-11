#pragma once

#include "einsums/Tensor.hpp"
#include "einsums/_Export.hpp"

namespace einsums::fft {

namespace detail {}

auto EINSUMS_EXPORT fftfreq(int n, double d = 1.0) -> Tensor<double, 1>;

namespace detail {

/*************************************
 * Real or complex -> complex        *
 *************************************/
void EINSUMS_EXPORT scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result);

void EINSUMS_EXPORT dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result);
void EINSUMS_EXPORT zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result);

/*************************************
 * Real or complex <- complex        *
 *************************************/
void EINSUMS_EXPORT csifft(const Tensor<std::complex<float>, 1> &a, Tensor<float, 1> *result);
void EINSUMS_EXPORT zdifft(const Tensor<std::complex<double>, 1> &a, Tensor<double, 1> *result);

void EINSUMS_EXPORT ccifft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT zzifft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result);

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

inline void ifft(const Tensor<std::complex<float>, 1> &a, Tensor<float, 1> *result) {
    detail::csifft(a, result);
}

inline void ifft(const Tensor<std::complex<double>, 1> &a, Tensor<double, 1> *result) {
    detail::zdifft(a, result);
}

inline void ifft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    detail::ccifft(a, result);
}

inline void ifft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    detail::zzifft(a, result);
}

} // namespace einsums::fft