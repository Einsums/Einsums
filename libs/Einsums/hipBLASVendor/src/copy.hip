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

void scopy(int64_t n, float const *x, int64_t inc_x, float *y, int64_t inc_y) {
    LabeledSection("scopy");
    hipblas_catch(hipblasScopy_64(gpu::get_blas_handle(), n, x, inc_x, y, inc_y));
    gpu::stream_wait();
}

void dcopy(int64_t n, double const *x, int64_t inc_x, double *y, int64_t inc_y) {
    LabeledSection("dcopy");
    hipblas_catch(hipblasDcopy_64(gpu::get_blas_handle(), n, x, inc_x, y, inc_y));
    gpu::stream_wait();
}

void ccopy(int64_t n, std::complex<float> const *x, int64_t inc_x, std::complex<float> *y, int64_t inc_y) {
    LabeledSection("ccopy");
#if hipblasVersionMajor < 3
    hipblas_catch(hipblasCcopy_64(gpu::get_blas_handle(), n, (hipblasComplex const *)x, inc_x, (hipblasComplex *)y, inc_y));
#else
    hipblas_catch(hipblasCcopy_64(gpu::get_blas_handle(), n, (hipComplex const *)x, inc_x, (hipComplex *)y, inc_y));
#endif
    gpu::stream_wait();
}

void zcopy(int64_t n, std::complex<double> const *x, int64_t inc_x, std::complex<double> *y, int64_t inc_y) {
    LabeledSection("zcopy");
#if hipblasVersionMajor < 3
    hipblas_catch(hipblasZcopy_64(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, inc_x, (hipblasDoubleComplex *)y, inc_y));
#else
    hipblas_catch(hipblasZcopy_64(gpu::get_blas_handle(), n, (hipDoubleComplex const *)x, inc_x, (hipDoubleComplex *)y, inc_y));
#endif
    gpu::stream_wait();
}

} // namespace einsums::blas::hip