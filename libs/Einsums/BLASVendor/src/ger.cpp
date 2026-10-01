//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

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

namespace {
void ger_parameter_check(int_t m, int_t n, int_t inc_x, int_t inc_y, int_t lda) {
    char buf[256] = {0};
    if (m < 0) {
        fmt::format_to(buf, "einsums::backend::vendor::ger: m ({}) is less than zero.", m);
        throw std::runtime_error(buf);
    }
    if (n < 0) {
        fmt::format_to(buf, "einsums::backend::vendor::ger: n ({}) is less than zero.", n);
        throw std::runtime_error(buf);
    }
    if (inc_x == 0) {
        fmt::format_to(buf, "einsums::backend::vendor::ger: inc_x ({}) is zero.", inc_x);
        throw std::runtime_error(buf);
    }
    if (inc_y == 0) {
        fmt::format_to(buf, "einsums::backend::vendor::ger: inc_y ({}) is zero.", inc_y);
        throw std::runtime_error(buf);
    }
    if (lda < std::max(int_t{1}, n)) {
        fmt::format_to(buf, "einsums::backend::vendor::ger: lda ({}) is less than max(1, n ({})).", lda, n);
        throw std::runtime_error(buf);
    }
}
} // namespace

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