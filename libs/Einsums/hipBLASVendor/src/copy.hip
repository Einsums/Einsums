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

void scopy(int n, float const *x, int inc_x, float *y, int inc_y) {
    LabeledSection("scopy");
    hipblas_catch(hipblasScopy(gpu::get_blas_handle(), n, x, inc_x, y, inc_y));
    gpu::stream_wait();
}

void dcopy(int n, double const *x, int inc_x, double *y, int inc_y) {
    LabeledSection("dcopy");
    hipblas_catch(hipblasDcopy(gpu::get_blas_handle(), n, x, inc_x, y, inc_y));
    gpu::stream_wait();
}

void ccopy(int n, std::complex<float> const *x, int inc_x, std::complex<float> *y, int inc_y) {
    LabeledSection("ccopy");
    hipblas_catch(hipblasCcopy(gpu::get_blas_handle(), n, (hipblasComplex const *)x, inc_x, (hipblasComplex *)y, inc_y));
    gpu::stream_wait();
}

void zcopy(int n, std::complex<double> const *x, int inc_x, std::complex<double> *y, int inc_y) {
    LabeledSection("zcopy");
    hipblas_catch(hipblasZcopy(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, inc_x, (hipblasDoubleComplex *)y, inc_y));
    gpu::stream_wait();
}

} // namespace einsums::blas::hip