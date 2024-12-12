//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/FFT.hpp>
#include <Einsums/FFT/Defines.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/ARange.hpp>

#if defined(EINSUMS_HAVE_FFT_LIBRARY_FFTW3)
#    include "Backends/FFTW3/FFT.hpp"
#endif

namespace einsums::fft {

auto fftfreq(int n, double d) -> Tensor<double, 1> {
    double value   = 1.0 / (n * d);
    auto   results = create_tensor("FFTFreq", n);

    int  N               = (n - 1) / 2 + 1;
    auto p1              = arange<double>(0, N);
    results(Range(0, N)) = p1;

    auto p2              = arange<double>(-(n / 2), 0);
    results(Range(N, n)) = p2;

    linear_algebra::scale(value, &results);

    return results;
}

namespace detail {

namespace {
template <typename T>
void check_size(Tensor<T, 1> const &a, Tensor<std::complex<T>, 1> const *result) {
    if (result->dim(0) >= a.dim(0) / 2 + 1)
        return;

    println_abort("fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}", a.name(), a.dim(0),
                  result->name(), result->dim(0));
}

template <typename T>
void icheck_size(Tensor<std::complex<T>, 1> const &a, Tensor<T, 1> const *result) {
    if (a.dim(0) >= result->dim(0) / 2 + 1)
        return;

    println_abort("ifft called with too small");
}
} // namespace

void scfft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    check_size(a, result);
    // backend::mkl::scfft(a, result);
    backend::fftw3::scfft(a, result);
}

void ccfft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    // backend::mkl::ccfft(a, result);
    backend::fftw3::ccfft(a, result);
}

void dzfft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    check_size(a, result);
    // backend::mkl::dzfft(a, result);
    backend::fftw3::dzfft(a, result);
}

void zzfft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    // backend::mkl::zzfft(a, result);
    backend::fftw3::zzfft(a, result);
}

void csifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::csifft(a, result);
    backend::fftw3::csifft(a, result);
}

void zdifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::zdifft(a, result);
    backend::fftw3::zdifft(a, result);
}

void ccifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 1> *result) {
    // TODO: Add appropriate icheck_size(...);
    // backend::mkl::ccifft(a, result);
    backend::fftw3::ccifft(a, result);
}

void zzifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 1> *result) {
    // TODO: Add appropriate icheck_size(...);
    // backend::mkl::zzifft(a, result);
    backend::fftw3::zzifft(a, result);
}

} // namespace detail

} // namespace einsums::fft