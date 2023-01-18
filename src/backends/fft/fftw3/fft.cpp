#include "fft.hpp"

#include "einsums/Print.hpp"

#if defined(EINSUMS_HAVE_FFT_LIBRARY_MKL)
#include <fftw/fftw3.h>
#elif defined(EINSUMS_HAVE_FFT_LIBRARY_FFTW3)
#include <fftw3.h>
#else
#error Unable to find FFTW header.
#endif

namespace einsums::backend::fftw3 {

namespace {
template <typename Plan>
inline void verify(Plan plan) {
    if (plan == nullptr) {
        println_abort("einsums::backend::fftw3::XXfft: Unable to create FFTW plan.");
    }
}
} // namespace

/*******************************************************************************
 * Forward transforms                                                          *
 *******************************************************************************/

void scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result) {
    fftwf_plan r2c = nullptr;

    verify((r2c = fftwf_plan_dft_r2c_1d(a.dim(0), const_cast<float *>(a.data()), reinterpret_cast<fftwf_complex *>(result->data()),
                                        FFTW_ESTIMATE)));
    fftwf_execute(r2c);
    fftwf_destroy_plan(r2c);
}

void ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    fftwf_plan c2c = nullptr;

    verify((c2c = fftwf_plan_dft_1d(a.dim(0), reinterpret_cast<fftwf_complex *>(const_cast<std::complex<float> *>(a.data())),
                                    reinterpret_cast<fftwf_complex *>(result->data()), FFTW_FORWARD, FFTW_ESTIMATE)));
    fftwf_execute(c2c);
    fftwf_destroy_plan(c2c);
}

void dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result) {
    fftw_plan r2c = nullptr;

    verify((r2c = fftw_plan_dft_r2c_1d(a.dim(0), const_cast<double *>(a.data()), reinterpret_cast<fftw_complex *>(result->data()),
                                       FFTW_ESTIMATE)));
    fftw_execute(r2c);
    fftw_destroy_plan(r2c);
}

void zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    fftw_plan c2c = nullptr;

    verify((c2c = fftw_plan_dft_1d(a.dim(0), reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(a.data())),
                                   reinterpret_cast<fftw_complex *>(result->data()), FFTW_FORWARD, FFTW_ESTIMATE)));
    fftw_execute(c2c);
    fftw_destroy_plan(c2c);
}

/*******************************************************************************
 * Backward transforms                                                         *
 *******************************************************************************/
void csifft(const Tensor<std::complex<float>, 1> &a, Tensor<float, 1> *result) {
    fftwf_plan c2r = nullptr;

    verify(c2r = fftwf_plan_dft_c2r_1d(result->dim(0), reinterpret_cast<fftwf_complex *>(const_cast<std::complex<float> *>(a.data())),
                                       result->data(), FFTW_ESTIMATE));
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);
}

void zdifft(const Tensor<std::complex<double>, 1> &a, Tensor<double, 1> *result) {
    fftw_plan c2r = nullptr;

    verify(c2r = fftw_plan_dft_c2r_1d(result->dim(0), reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(a.data())),
                                      result->data(), FFTW_ESTIMATE));
    fftw_execute(c2r);
    fftw_destroy_plan(c2r);
}

void ccifft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result) {
    fftwf_plan c2r = nullptr;

    verify(c2r = fftwf_plan_dft_1d(result->dim(0), reinterpret_cast<fftwf_complex *>(const_cast<std::complex<float> *>(a.data())),
                                   reinterpret_cast<fftwf_complex *>(result->data()), FFTW_BACKWARD, FFTW_ESTIMATE));
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);
}

void zzifft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result) {
    fftw_plan c2c = nullptr;

    verify(c2c = fftw_plan_dft_1d(result->dim(0), reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(a.data())),
                                  reinterpret_cast<fftw_complex *>(result->data()), FFTW_BACKWARD, FFTW_ESTIMATE));
    fftw_execute(c2c);
    fftw_destroy_plan(c2c);
}

} // namespace einsums::backend::fftw3