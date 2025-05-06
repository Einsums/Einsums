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
extern void FC_GLOBAL(sscal, SSCAL)(int_t *, float *, float *, int_t *);
extern void FC_GLOBAL(dscal, DSCAL)(int_t *, double *, double *, int_t *);
extern void FC_GLOBAL(cscal, CSCAL)(int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zscal, ZSCAL)(int_t *, std::complex<double> *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(csscal, CSSCAL)(int_t *, float *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zdscal, ZDSCAL)(int_t *, double *, std::complex<double> *, int_t *);
}

void sscal(int_t n, float alpha, float *vec, int_t inc) {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    FC_GLOBAL(sscal, SSCAL)(&n, &alpha, vec, &inc);
}

void dscal(int_t n, double alpha, double *vec, int_t inc) {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    FC_GLOBAL(dscal, DSCAL)(&n, &alpha, vec, &inc);
}

void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc) {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    FC_GLOBAL(cscal, CSCAL)(&n, &alpha, vec, &inc);
}

void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc) {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    FC_GLOBAL(zscal, ZSCAL)(&n, &alpha, vec, &inc);
}

void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    FC_GLOBAL(csscal, CSSCAL)(&n, &alpha, vec, &inc);
}

void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    FC_GLOBAL(zdscal, ZDSCAL)(&n, &alpha, vec, &inc);
}

} // namespace einsums::blas::vendor