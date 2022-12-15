#include "einsums/FFT.hpp"

#include "einsums/STL.hpp"

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

template <typename T>
void init_data(einsums::Tensor<std::complex<T>, 1> &data, int N, int H) {
    double TWOPI = 6.2831853071795864769, phase;
    int n;

    for (n = 0; n < N; n++) {
        phase = moda<T>(n, H, N) / N;
        data(n).real(cos(TWOPI * phase) / N);
        data(n).imag(sin(TWOPI * phase) / N);
    }
}

template <typename Source, typename Result, int N = 6, int H = -1>
void fft1d_1() {
    using namespace einsums;

    auto x_data = create_tensor<Source>("sample data", N);
    auto x_result = create_tensor<Result>("FFT result", einsums::is_complex_v<Source> ? N : N / 2 + 1);

    init_data(x_data, N, H);

    // println(x_data);

    fft::fft(x_data, &x_result);

    // The result of each forward transformation is a tensor with
    // exactly one 1.0 and everything else is 0.0.
    for (int n = 0; n < x_result.dim(0); n++) {
        einsums::remove_complex_t<Result> re_exp = 0.0, im_exp = 0.0;

        // The position of the expected 1.0 is dependent on the Source data type.
        if constexpr (!einsums::is_complex_v<Source>) {
            if ((n - H) % N == 0 || (-n - H) % N == 0)
                re_exp = 1.0;
        } else {
            if (n == x_result.dim(0) - 1)
                re_exp = 1.0;
        }

        REQUIRE_THAT(x_result(n).real(), Catch::Matchers::WithinAbsMatcher(einsums::remove_complex_t<Result>(re_exp), 0.0001));
        REQUIRE_THAT(x_result(n).imag(), Catch::Matchers::WithinAbsMatcher(einsums::remove_complex_t<Result>(im_exp), 0.0001));
    }
}

TEST_CASE("fft1") {
    fft1d_1<float, std::complex<float>>();
    fft1d_1<double, std::complex<double>>();
    fft1d_1<double, std::complex<double>, 10>();

    fft1d_1<std::complex<float>, std::complex<float>>();
    fft1d_1<std::complex<double>, std::complex<double>>();
}