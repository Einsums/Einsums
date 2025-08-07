//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <complex>

namespace einsums::linear_algebra::detail {
template void impl_direct_product<float, float, float>(float alpha, einsums::detail::TensorImpl<float> const &A,
                                                                             einsums::detail::TensorImpl<float> const &B, float beta,
                                                                             einsums::detail::TensorImpl<float> *C);
template void impl_direct_product<double, double, double>(double alpha, einsums::detail::TensorImpl<double> const &A,
                                                                                einsums::detail::TensorImpl<double> const &B, double beta,
                                                                                einsums::detail::TensorImpl<double> *C);
template void impl_direct_product<std::complex<float>, std::complex<float>, std::complex<float>>(
    std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
template void impl_direct_product<std::complex<double>, std::complex<double>, std::complex<double>>(
    std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);
} // namespace einsums::linear_algebra::detail