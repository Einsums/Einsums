#include <cmath>
#include <complex>
namespace einsums {
namespace linear_algebra {
namespace detail {

constexpr float mult_error(float x, float y) {
    return std::fma(x, y, -x * y);
}

constexpr double mult_error(double x, double y) {
    return std::fma(x, y, -x * y);
}

constexpr float fma(float const &x, float const &y, float const &sum) {
    return std::fma(x, y, sum);
}

constexpr double fma(double const &x, double const &y, double const &sum) {
    return std::fma(x, y, sum);
}

constexpr std::complex<float> fma(std::complex<float> const &x, std::complex<float> const &y, std::complex<float> const &sum) {
    float real_real = std::real(x) * std::real(y);
    float imag_imag = std::imag(x) * std::imag(y);
    float real_imag = std::real(x) * std::imag(y);
    float imag_real = std::imag(x) * std::imag(y);

    float real = real_real - imag_imag;
    float imag = real_imag + imag_real;

    float real_real_error = mult_error(std::real(x), std::real(y));
    float imag_imag_error = mult_error(std::imag(x), std::imag(y));
    float real_imag_error = mult_error(std::real(x), std::imag(y));
    float imag_real_error = mult_error(std::imag(x), std::real(y));

    float real_error = real_real_error - imag_imag_error;
    float imag_error = real_imag_error + imag_real_error;

    return std::complex<float>{real + real_error, imag + imag_error};
}

constexpr std::complex<double> fma(std::complex<double> const &x, std::complex<double> const &y, std::complex<double> const &sum) {
    double real_real = std::real(x) * std::real(y);
    double imag_imag = std::imag(x) * std::imag(y);
    double real_imag = std::real(x) * std::imag(y);
    double imag_real = std::imag(x) * std::imag(y);

    double real = real_real - imag_imag;
    double imag = real_imag + imag_real;

    double real_real_error = mult_error(std::real(x), std::real(y));
    double imag_imag_error = mult_error(std::imag(x), std::imag(y));
    double real_imag_error = mult_error(std::real(x), std::imag(y));
    double imag_real_error = mult_error(std::imag(x), std::real(y));

    double real_error = real_real_error - imag_imag_error;
    double imag_error = real_imag_error + imag_real_error;

    return std::complex<double>{real + real_error, imag + imag_error};
}

constexpr std::complex<float> fma(std::complex<float> const &x, float const &y, std::complex<float> const &sum) {
    return std::complex<float>{std::fma(std::real(x), y, std::real(sum)), std::fma(std::imag(x), y, std::imag(sum))};
}

constexpr std::complex<double> fma(std::complex<double> const &x, double const &y, std::complex<double> const &sum) {
    return std::complex<double>{std::fma(std::real(x), y, std::real(sum)), std::fma(std::imag(x), y, std::imag(sum))};
}

constexpr std::complex<float> fma(float const &x, std::complex<float> const &y, std::complex<float> const &sum) {
    return fma(y, x, sum);
}

constexpr std::complex<double> fma(double const &x, std::complex<double> const &y, std::complex<double> const &sum) {
    return fma(y, x, sum);
}

template <typename T>
constexpr T triple_product(T const &x, T const &y, T const &z) {
    T first_prod  = x * y;
    T first_error = fma(x, y, -first_prod);

    T second_prod  = z * first_prod;
    T second_error = z * first_error;
    T third_error  = fma(z, first_prod, -second_prod);

    T error = second_error + third_error;

    return second_prod + error;
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums