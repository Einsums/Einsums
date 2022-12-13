#include "einsums/FFT.hpp"

#include "backends/fft/mkl/fft.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/STL.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"

namespace einsums::fft {

auto fftfreq(int n, double d) -> Tensor<double, 1> {
    double value = 1.0 / (n * d);
    auto results = create_tensor("FFTFreq", n);

    int N = (n - 1) / 2 + 1;
    auto p1 = arange<double>(0, N);
    results(Range(0, N)) = p1;

    auto p2 = arange<double>(-(n / 2), 0);
    results(Range(N, n)) = p2;

    linear_algebra::scale(value, &results);

    return results;
}

namespace detail {

void scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result) {
    backend::mkl::scfft(a, result);
}

void ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result);

void dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result) {
    backend::mkl::dzfft(a, result);
}

void zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result);
} // namespace detail

} // namespace einsums::fft