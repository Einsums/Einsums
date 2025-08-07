//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/LabeledSection.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(sger, DGER)(int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *, float *, int_t *);
extern void FC_GLOBAL(dger, DGER)(int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *, double *, int_t *);
extern void FC_GLOBAL(cgeru, CGERU)(int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgeru, ZGERU)(int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                    std::complex<double> const *, int_t *, std::complex<double> *, int_t *);
extern void FC_GLOBAL(cgerc, CGERC)(int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgerc, ZGERC)(int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                    std::complex<double> const *, int_t *, std::complex<double> *, int_t *);
}

#define ger_parameter_check(m, n, inc_x, inc_y, lda)                                                                                       \
    if ((m) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "First parameter (m) in ger call ({}) is less than zero.", m);                         \
    }                                                                                                                                      \
    if ((n) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Second parameter (n) in ger call ({}) is less than zero.", n);                        \
    }                                                                                                                                      \
    if ((inc_x) == 0) {                                                                                                                    \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Fifth parameter (inc_x) in ger call ({}) is zero.", inc_x);                           \
    }                                                                                                                                      \
    if ((inc_y) == 0) {                                                                                                                    \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Seventh parameter (inc_y) in ger call ({}) is zero.", inc_y);                         \
    }                                                                                                                                      \
    if ((lda) < std::max(int_t{1}, n)) {                                                                                                   \
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Ninth parameter (lda) in ger call ({}) is less than max(1, n ({})).", lda, n);        \
    }

void sger(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(sger, SGER)(&m, &n, &alpha, x, &inc_x, y, &inc_y, a, &lda);
}

void dger(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(dger, DGER)(&m, &n, &alpha, x, &inc_x, y, &inc_y, a, &lda);
}

void cger(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x, std::complex<float> const *y, int_t inc_y,
          std::complex<float> *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(cgeru, CGERU)(&m, &n, &alpha, x, &inc_x, y, &inc_y, a, &lda);
}

void zger(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x, std::complex<double> const *y,
          int_t inc_y, std::complex<double> *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    FC_GLOBAL(zgeru, ZGERU)(&m, &n, &alpha, x, &inc_x, y, &inc_y, a, &lda);
}

void cgerc(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x, std::complex<float> const *y,
           int_t inc_y, std::complex<float> *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(cgerc, CGERC)(&m, &n, &alpha, x, &inc_x, y, &inc_y, a, &lda);
}

void zgerc(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x, std::complex<double> const *y,
           int_t inc_y, std::complex<double> *a, int_t lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);

    FC_GLOBAL(zgerc, ZGERC)(&m, &n, &alpha, x, &inc_x, y, &inc_y, a, &lda);
}

} // namespace einsums::blas::vendor