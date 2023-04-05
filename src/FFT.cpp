#include "einsums/FFT.hpp"

#if defined(EINSUMS_HAVE_FFT_LIBRARY_FFTW3)
#    include "backends/fft/fftw3/fft.hpp"
#endif
// MKL provides both FFTW and Dfti interfaces
#if defined(EINSUMS_HAVE_FFT_LIBRARY_MKL)
#    include "backends/fft/fftw3/fft.hpp"
#    include "backends/fft/mkl/fft.hpp"
#endif
#include "einsums/LinearAlgebra.hpp"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"

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
void check_size(const Tensor<T, 1> &a, const Tensor<std::complex<T>, 1> *result) {
    if (result->dim(0) >= a.dim(0) / 2 + 1)
        return;

    println_abort("fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}", a.name(), a.dim(0),
                  result->name(), result->dim(0));
}

template <typename T>
void icheck_size(const Tensor<std::complex<T>, 1> &a, const Tensor<T, 1> *result) {
    if (a.dim(0) >= result->dim(0) / 2 + 1)
        return;

    println_abort("ifft called with too small");
}
} // namespace

void scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result) {
    check_size(a, result);
    // backend::mkl::scfft(a, result);
    backend::fft::fftw3::scfft(a, result);
}

void ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    // backend::mkl::ccfft(a, result);
    backend::fft::fftw3::ccfft(a, result);
}

void dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result) {
    check_size(a, result);
    // backend::mkl::dzfft(a, result);
    backend::fft::fftw3::dzfft(a, result);
}

void zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    // backend::mkl::zzfft(a, result);
    backend::fft::fftw3::zzfft(a, result);
}

void csifft(const Tensor<std::complex<float>, 1> &a, Tensor<float, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::csifft(a, result);
    backend::fft::fftw3::csifft(a, result);
}

void zdifft(const Tensor<std::complex<double>, 1> &a, Tensor<double, 1> *result) {
    icheck_size(a, result);
    // backend::mkl::zdifft(a, result);
    backend::fft::fftw3::zdifft(a, result);
}

void ccifft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    // TODO: Add appropriate icheck_size(...);
    // backend::mkl::ccifft(a, result);
    backend::fft::fftw3::ccifft(a, result);
}

void zzifft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    // TODO: Add appropriate icheck_size(...);
    // backend::mkl::zzifft(a, result);
    backend::fft::fftw3::zzifft(a, result);
}

} // namespace detail

} // namespace einsums::fft