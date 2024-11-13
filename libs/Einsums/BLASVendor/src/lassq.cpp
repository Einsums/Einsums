//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/LabeledSection.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(slassq, SLASSQ)(int_t *n, float const *x, int_t *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(dlassq, DLASSQ)(int_t *n, double const *x, int_t *incx, double *scale, double *sumsq);
extern void FC_GLOBAL(classq, CLASSQ)(int_t *n, std::complex<float> const *x, int_t *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(zlassq, ZLASSQ)(int_t *n, std::complex<double> const *x, int_t *incx, double *scale, double *sumsq);
}

void slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(slassq, SLASSQ)(&n, x, &incx, scale, sumsq);
}

void dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(dlassq, DLASSQ)(&n, x, &incx, scale, sumsq);
}

void classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(classq, CLASSQ)(&n, x, &incx, scale, sumsq);
}

void zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(zlassq, ZLASSQ)(&n, x, &incx, scale, sumsq);
}

} // namespace einsums::blas::vendor