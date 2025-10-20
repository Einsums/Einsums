//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <exception>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {

void clacgv(int64_t n, std::complex<float> *x, int64_t incx) {
    LabeledSection0();

    float *x_imag = (float *)x;
    x_imag++; // This is the imaginary components of the input vector.

    sscal(n, -1.0f, x_imag, 2 * incx);
}

void zlacgv(int64_t n, std::complex<double> *x, int64_t incx) {
    LabeledSection0();

    double *x_imag = (double *)x;
    x_imag++; // This is the imaginary components of the input vector.

    dscal(n, -1.0, x_imag, 2 * incx);
}

} // namespace einsums::blas::hip