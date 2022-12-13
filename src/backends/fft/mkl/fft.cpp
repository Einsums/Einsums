#include "fft.hpp"

#include <mkl_dfti.h>

namespace einsums::backend::mkl {

void scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result) {
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    MKL_LONG status;

    status = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, a.dim(0));
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(handle);

    auto *x_real = const_cast<float *>(a.data());
    auto *x_cmplx = (MKL_Complex8 *)result->data();

    status = DftiComputeForward(handle, x_real, x_cmplx);

    DftiFreeDescriptor(&handle);
}

void dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result) {
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    MKL_LONG status;

    status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, a.dim(0));
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(handle);

    auto *x_real = const_cast<double *>(a.data());
    auto *x_cmplx = (MKL_Complex16 *)result->data();

    status = DftiComputeForward(handle, x_real, x_cmplx);

    DftiFreeDescriptor(&handle);
}
} // namespace einsums::backend::mkl