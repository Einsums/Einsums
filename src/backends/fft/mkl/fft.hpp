#pragma once

#include "einsums/Tensor.hpp"

namespace einsums::backend::mkl {

void scfft(const Tensor<float, 1> &a, Tensor<std::complex<float>, 1> *result);
void ccfft(const Tensor<std::complex<float>, 1> &a, Tensor<std::complex<float>, 1> *result);

void dzfft(const Tensor<double, 1> &a, Tensor<std::complex<double>, 1> *result);
void zzfft(const Tensor<std::complex<double>, 1> &a, Tensor<std::complex<double>, 1> *result);

} // namespace einsums::backend::mkl