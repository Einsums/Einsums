//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>
#include <source_location>

#include "Common.hpp"

namespace einsums::blas::hip {

void scopy(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    LabeledSection("scopy");
    hipblas_catch(hipblasScopy(gpu::get_blas_handle(), n, x, inc_x, y, inc_y));
}

void dcopy(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    LabeledSection("dcopy");
    hipblas_catch(hipblasDcopy(gpu::get_blas_handle(), n, x, inc_x, y, inc_y));
}

void ccopy(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    LabeledSection("ccopy");
    hipblas_catch(hipblasCcopy(gpu::get_blas_handle(), n, (hipblasComplex const *)x, inc_x, (hipblasComplex *)y, inc_y));
}

void zcopy(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    LabeledSection("zcopy");
    hipblas_catch(hipblasZcopy(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, inc_x, (hipblasDoubleComplex *)y, inc_y));
}

} // namespace einsums::blas::hip