/*
 * Copyright (c) 2022 Justin Turney
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file FFT.hpp
 *
 * Contains things for performing fast Fourier transforms.
 */

#pragma once

#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

namespace einsums::fft {

namespace detail {}

/**
 * Find the fast Fourier frequencies.
 * 
 * @param n The number of items in the list.
 * @param d The scale factor.
 *
 * @return The frequency step in the list.
 */
auto EINSUMS_EXPORT fftfreq(int n, double d = 1.0) -> Tensor<double, 1>;

namespace detail {

/*************************************
 * Real or complex -> complex        *
 *************************************/
void EINSUMS_EXPORT scfft(const Tensor<float, 1> &a,
                          Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT ccfft(const Tensor<std::complex<float>, 1> &a,
                          Tensor<std::complex<float>, 1> *result);

void EINSUMS_EXPORT dzfft(const Tensor<double, 1> &a,
                          Tensor<std::complex<double>, 1> *result);
void EINSUMS_EXPORT zzfft(const Tensor<std::complex<double>, 1> &a,
                          Tensor<std::complex<double>, 1> *result);

/*************************************
 * Real or complex <- complex        *
 *************************************/
void EINSUMS_EXPORT csifft(const Tensor<std::complex<float>, 1> &a,
                           Tensor<float, 1> *result);
void EINSUMS_EXPORT zdifft(const Tensor<std::complex<double>, 1> &a,
                           Tensor<double, 1> *result);

void EINSUMS_EXPORT ccifft(const Tensor<std::complex<float>, 1> &a,
                           Tensor<std::complex<float>, 1> *result);
void EINSUMS_EXPORT zzifft(const Tensor<std::complex<double>, 1> &a,
                           Tensor<std::complex<double>, 1> *result);

}  // namespace detail

inline void fft(const Tensor<float, 1> &a,
                Tensor<std::complex<float>, 1> *result) {
    detail::scfft(a, result);
}

inline void fft(const Tensor<std::complex<float>, 1> &a,
                Tensor<std::complex<float>, 1> *result) {
    detail::ccfft(a, result);
}

inline void fft(const Tensor<double, 1> &a,
                Tensor<std::complex<double>, 1> *result) {
    detail::dzfft(a, result);
}

inline void fft(const Tensor<std::complex<double>, 1> &a,
                Tensor<std::complex<double>, 1> *result) {
    detail::zzfft(a, result);
}

inline void ifft(const Tensor<std::complex<float>, 1> &a,
                 Tensor<float, 1> *result) {
    detail::csifft(a, result);
}

inline void ifft(const Tensor<std::complex<double>, 1> &a,
                 Tensor<double, 1> *result) {
    detail::zdifft(a, result);
}

inline void ifft(const Tensor<std::complex<float>, 1> &a,
                 Tensor<std::complex<float>, 1> *result) {
    detail::ccifft(a, result);
}

inline void ifft(const Tensor<std::complex<double>, 1> &a,
                 Tensor<std::complex<double>, 1> *result) {
    detail::zzifft(a, result);
}

}  // namespace einsums::fft
