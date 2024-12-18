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
extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, int_t *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *,
                                    float *, float *, int_t *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int_t *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *,
                                    double *, double *, int_t *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *,
                                    int_t *, std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *,
                                    int_t *);
}

void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
           float beta, float *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(sgemm, SGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
           double beta, double *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(dgemm, DGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(cgemm, CGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(zgemm, ZGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

} // namespace einsums::blas::vendor