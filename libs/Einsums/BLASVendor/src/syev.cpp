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
extern void FC_GLOBAL(ssyev, SSYEV)(char *, char *, int_t *, float *, int_t *, float *, float *, int_t *, int_t *);
extern void FC_GLOBAL(dsyev, DSYEV)(char *, char *, int_t *, double *, int_t *, double *, double *, int_t *, int_t *);
}

auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    int_t info{0};
    FC_GLOBAL(ssyev, SSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    EINSUMS_PROFILE_SCOPE("BLASVendor");
    ;

    int_t info{0};
    FC_GLOBAL(dsyev, DSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

} // namespace einsums::blas::vendor