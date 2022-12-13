#include "einsums/FFT.hpp"

#include <catch2/catch.hpp>

// Some elements of these tests come from Intel MKL
// examples/c/dft/sources/basic_dp_real_dft_1d.c

template <typename T>
auto moda(int K, int L, int M) -> T {
    return (T)(((long long)K * L) % M);
}

template <typename T>
void init_data(einsums::Tensor<T, 1> &data, int N, int H) {
    T TWOPI = 6.2831853071795864769, phase, factor;
    int n;

    factor = (2 * (N - H) % N == 0) ? 1.0 : 2.0;
    for (n = 0; n < N; n++) {
        phase = moda<T>(n, H, N) / N;
        data(n) = factor * cos(TWOPI * phase) / N;
    }
}

template <typename Source, typename Result, int N = 6, int H = -1>
void fft1d_1() {
    using namespace einsums;

    auto x_data = create_tensor<Source>("sample data", N);
    auto x_result = create_tensor<Result>("FFT result", N / 2 + 1);

    init_data(x_data, N, H);

    println(x_data);

    fft::fft(x_data, &x_result);

    // Perform checks on the result.
    println(x_result);
}

TEST_CASE("fft1") {
    fft1d_1<float, std::complex<float>>();
    fft1d_1<double, std::complex<double>>();

    fft1d_1<double, std::complex<double>, 10>();
}