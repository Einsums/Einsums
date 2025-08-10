//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(scopy, SCOPY)(int_t *, float const *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dcopy, DCOPY)(int_t *, double const *, int_t *, double *, int_t *);
extern void FC_GLOBAL(ccopy, CCOPY)(int_t *, std::complex<float> const *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zcopy, ZCOPY)(int_t *, std::complex<double> const *, int_t *, std::complex<double> *, int_t *);
}

void scopy(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    LabeledSection("scopy");
    FC_GLOBAL(scopy, SCOPY)(&n, x, &inc_x, y, &inc_y);
}

void dcopy(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    LabeledSection("dcopy");
    FC_GLOBAL(dcopy, DCOPY)(&n, x, &inc_x, y, &inc_y);
}

void ccopy(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    LabeledSection("ccopy");
    FC_GLOBAL(ccopy, CCOPY)(&n, x, &inc_x, y, &inc_y);
}

void zcopy(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    LabeledSection("zcopy");
    FC_GLOBAL(zcopy, ZCOPY)(&n, x, &inc_x, y, &inc_y);
}

} // namespace einsums::blas::vendor