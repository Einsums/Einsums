#include "einsums/FFT.hpp"

#include "backends/fft/mkl/fft.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/Print.hpp"
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

namespace {
template <typename T>
void check_size(const Tensor<T, 1> &a, const Tensor<std::complex<T>, 1> *result) {
    if (result->dim(0) >= a.dim(0) / 2 + 1)
        return;

    println_abort("fft called with too small result tensor size\nsize of \"{}\" is {}\nsize of \"{}\" is {}", a.name(), a.dim(0),
                  result->name(), result->dim(0));
}
} // namespace

void scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result) {
    check_size(a, result);
    backend::mkl::scfft(a, result);
}

void ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    backend::mkl::ccfft(a, result);
}

void dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result) {
    check_size(a, result);
    backend::mkl::dzfft(a, result);
}

void zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    backend::mkl::zzfft(a, result);
}
} // namespace detail

} // namespace einsums::fft