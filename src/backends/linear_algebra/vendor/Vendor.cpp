//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Vendor.hpp"

#include "einsums/_Common.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Print.hpp"
#include "einsums/Section.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "Utilities.hpp"

#ifndef FC_SYMBOL
#    define FC_SYMBOL 2
#endif

#if FC_SYMBOL == 1
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) name
#elif FC_SYMBOL == 2
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) name##_
#elif FC_SYMBOL == 3
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) NAME
#elif FC_SYMBOL == 4
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) NAME##_
#endif

extern "C" {

extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, int *, int *, int *, float *, const float *, int *, const float *, int *, float *,
                                    float *, int *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int *, int *, int *, double *, const double *, int *, const double *, int *, double *,
                                    double *, int *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, int *, int *, int *, std::complex<float> *, const std::complex<float> *, int *,
                                    const std::complex<float> *, int *, std::complex<float> *, std::complex<float> *, int *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, int *, int *, int *, std::complex<double> *, const std::complex<double> *, int *,
                                    const std::complex<double> *, int *, std::complex<double> *, std::complex<double> *, int *);

extern void FC_GLOBAL(sgemv, SGEMV)(char *, int *, int *, float *, const float *, int *, const float *, int *, float *, float *, int *);
extern void FC_GLOBAL(dgemv, DGEMV)(char *, int *, int *, double *, const double *, int *, const double *, int *, double *, double *,
                                    int *);
extern void FC_GLOBAL(cgemv, CGEMV)(char *, int *, int *, std::complex<float> *, const std::complex<float> *, int *,
                                    const std::complex<float> *, int *, std::complex<float> *, std::complex<float> *, int *);
extern void FC_GLOBAL(zgemv, ZGEMV)(char *, int *, int *, std::complex<double> *, const std::complex<double> *, int *,
                                    const std::complex<double> *, int *, std::complex<double> *, std::complex<double> *, int *);

extern void FC_GLOBAL(cheev, CHEEV)(char *job, char *uplo, int *n, std::complex<float> *a, int *lda, float *w, std::complex<float> *work,
                                    int *lwork, float *rwork, int *info);
extern void FC_GLOBAL(zheev, ZHEEV)(char *job, char *uplo, int *n, std::complex<double> *a, int *lda, double *w, std::complex<double> *work,
                                    int *lwork, double *rwork, int *info);

extern void FC_GLOBAL(ssyev, SSYEV)(char *, char *, int *, float *, int *, float *, float *, int *, int *);
extern void FC_GLOBAL(dsyev, DSYEV)(char *, char *, int *, double *, int *, double *, double *, int *, int *);

extern void FC_GLOBAL(sgesv, SGESV)(int *, int *, float *, int *, int *, float *, int *, int *);
extern void FC_GLOBAL(dgesv, DGESV)(int *, int *, double *, int *, int *, double *, int *, int *);
extern void FC_GLOBAL(cgesv, CGESV)(int *, int *, std::complex<float> *, int *, int *, std::complex<float> *, int *, int *);
extern void FC_GLOBAL(zgesv, ZGESV)(int *, int *, std::complex<double> *, int *, int *, std::complex<double> *, int *, int *);

extern void FC_GLOBAL(sscal, SSCAL)(int *, float *, float *, int *);
extern void FC_GLOBAL(dscal, DSCAL)(int *, double *, double *, int *);
extern void FC_GLOBAL(cscal, CSCAL)(int *, std::complex<float> *, std::complex<float> *, int *);
extern void FC_GLOBAL(zscal, ZSCAL)(int *, std::complex<double> *, std::complex<double> *, int *);
extern void FC_GLOBAL(csscal, CSSCAL)(int *, float *, std::complex<float> *, int *);
extern void FC_GLOBAL(zdscal, ZDSCAL)(int *, double *, std::complex<double> *, int *);

extern float                FC_GLOBAL(sdot, SDOT)(int *, const float *, int *, const float *, int *);
extern double               FC_GLOBAL(ddot, DDOT)(int *, const double *, int *, const double *, int *);
extern std::complex<float>  FC_GLOBAL(cdotu, CDOTU)(int *, const std::complex<float> *, int *, const std::complex<float> *, int *);
extern std::complex<double> FC_GLOBAL(zdotu, ZDOTU)(int *, const std::complex<double> *, int *, const std::complex<double> *, int *);

extern void FC_GLOBAL(saxpy, SAXPY)(int *, float *, const float *, int *, float *, int *);
extern void FC_GLOBAL(daxpy, DAXPY)(int *, double *, const double *, int *, double *, int *);
extern void FC_GLOBAL(caxpy, CAXPY)(int *, std::complex<float> *, const std::complex<float> *, int *, std::complex<float> *, int *);
extern void FC_GLOBAL(zaxpy, ZAXPY)(int *, std::complex<double> *, const std::complex<double> *, int *, std::complex<double> *, int *);

extern void FC_GLOBAL(sger, DGER)(int *, int *, float *, const float *, int *, const float *, int *, float *, int *);
extern void FC_GLOBAL(dger, DGER)(int *, int *, double *, const double *, int *, const double *, int *, double *, int *);
extern void FC_GLOBAL(cgeru, CGERU)(int *, int *, std::complex<float> *, const std::complex<float> *, int *, const std::complex<float> *,
                                    int *, std::complex<float> *, int *);
extern void FC_GLOBAL(zgeru, ZGERU)(int *, int *, std::complex<double> *, const std::complex<double> *, int *, const std::complex<double> *,
                                    int *, std::complex<double> *, int *);

extern void FC_GLOBAL(sgetrf, SGETRF)(int *, int *, float *, int *, int *, int *);
extern void FC_GLOBAL(dgetrf, DGETRF)(int *, int *, double *, int *, int *, int *);
extern void FC_GLOBAL(cgetrf, CGETRF)(int *, int *, std::complex<float> *, int *, int *, int *);
extern void FC_GLOBAL(zgetrf, ZGETRF)(int *, int *, std::complex<double> *, int *, int *, int *);

extern void FC_GLOBAL(sgetri, SGETRI)(int *, float *, int *, int *, float *, int *, int *);
extern void FC_GLOBAL(dgetri, DGETRI)(int *, double *, int *, int *, double *, int *, int *);
extern void FC_GLOBAL(cgetri, CGETRI)(int *, std::complex<float> *, int *, int *, std::complex<float> *, int *, int *);
extern void FC_GLOBAL(zgetri, ZGETRI)(int *, std::complex<double> *, int *, int *, std::complex<double> *, int *, int *);

extern float  FC_GLOBAL(slange, SLANGE)(char, int, int, const float *, int, float *);                 // NOLINT
extern double FC_GLOBAL(dlange, DLANGE)(char, int, int, const double *, int, double *);               // NOLINT
extern float  FC_GLOBAL(clange, CLANGE)(char, int, int, const std::complex<float> *, int, float *);   // NOLINT
extern double FC_GLOBAL(zlange, ZLANGE)(char, int, int, const std::complex<double> *, int, double *); // NOLINT

extern void FC_GLOBAL(slassq, SLASSQ)(int *n, const float *x, int *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(dlassq, DLASSQ)(int *n, const double *x, int *incx, double *scale, double *sumsq);
extern void FC_GLOBAL(classq, CLASSQ)(int *n, const std::complex<float> *x, int *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(zlassq, ZLASSQ)(int *n, const std::complex<double> *x, int *incx, double *scale, double *sumsq);

extern void FC_GLOBAL(dgesdd, DGESDD)(char *, int *, int *, double *, int *, double *, double *, int *, double *, int *, double *, int *,
                                      int *, int *);
}

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::backend::linear_algebra::vendor)

void initialize() {
}
void finalize() {
}

void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(sgemm, SGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(dgemm, DGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(cgemm, CGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(zgemm, ZGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(sgemv, SGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(dgemv, DGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(cgemv, CGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    FC_GLOBAL(zgemv, ZGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(ssyev, SSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(dsyev, DSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *rwork)
    -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(cheev, CHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rwork)
    -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(zheev, ZHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}

auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(sgesv, SGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(dgesv, DGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(cgesv, CGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(zgesv, ZGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

void sscal(int n, float alpha, float *vec, int inc) {
    LabeledSection0();

    FC_GLOBAL(sscal, SSCAL)(&n, &alpha, vec, &inc);
}

void dscal(int n, double alpha, double *vec, int inc) {
    LabeledSection0();

    FC_GLOBAL(dscal, DSCAL)(&n, &alpha, vec, &inc);
}

void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc) {
    LabeledSection0();

    FC_GLOBAL(cscal, CSCAL)(&n, &alpha, vec, &inc);
}

void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc) {
    LabeledSection0();

    FC_GLOBAL(zscal, ZSCAL)(&n, &alpha, vec, &inc);
}

void csscal(int n, float alpha, std::complex<float> *vec, int inc) {
    LabeledSection0();

    FC_GLOBAL(csscal, CSSCAL)(&n, &alpha, vec, &inc);
}

void zdscal(int n, double alpha, std::complex<double> *vec, int inc) {
    LabeledSection0();

    FC_GLOBAL(zdscal, ZDSCAL)(&n, &alpha, vec, &inc);
}

auto sdot(int n, const float *x, int incx, const float *y, int incy) -> float {
    LabeledSection0();

    return FC_GLOBAL(sdot, SDOT)(&n, x, &incx, y, &incy);
}

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double {
    LabeledSection0();

    return FC_GLOBAL(ddot, DDOT)(&n, x, &incx, y, &incy);
}

auto cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float> {
    LabeledSection0();

    return FC_GLOBAL(cdotu, CDOTU)(&n, x, &incx, y, &incy);
}

auto zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double> {
    LabeledSection0();

    return FC_GLOBAL(zdotu, ZDOTU)(&n, x, &incx, y, &incy);
}

void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y) {
    LabeledSection0();

    FC_GLOBAL(saxpy, SAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    LabeledSection0();

    FC_GLOBAL(daxpy, DAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y) {
    LabeledSection0();

    FC_GLOBAL(caxpy, CAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y) {
    LabeledSection0();

    FC_GLOBAL(zaxpy, ZAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void saxpby(const int n, const float a, const float *x, const int incx, const float b, float *y, const int incy) {
    LabeledSection0();
    sscal(n, b, y, incy);
    saxpy(n, a, x, incx, y, incy);
}

void daxpby(const int n, const double a, const double *x, const int incx, const double b, double *y, const int incy) {
    LabeledSection0();
    dscal(n, b, y, incy);
    daxpy(n, a, x, incx, y, incy);
}

void caxpby(const int n, const std::complex<float> a, const std::complex<float> *x, const int incx, const std::complex<float> b,
            std::complex<float> *y, const int incy) {
    LabeledSection0();
    cscal(n, b, y, incy);
    caxpy(n, a, x, incx, y, incy);
}

void zaxpby(const int n, const std::complex<double> a, const std::complex<double> *x, const int incx, const std::complex<double> b,
            std::complex<double> *y, const int incy) {
    LabeledSection0();
    zscal(n, b, y, incy);
    zaxpy(n, a, x, incx, y, incy);
}

namespace {
void ger_parameter_check(int m, int n, int inc_x, int inc_y, int lda) {
    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: m ({}) is less than zero.", m));
    } else if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: n ({}) is less than zero.", n));
    } else if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: inc_x ({}) is zero.", inc_x));
    } else if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: inc_y ({}) is zero.", inc_y));
    } else if (lda < std::max(1, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: lda ({}) is less than max(1, n ({})).", lda, n));
    }
}
} // namespace

void sger(int m, int n, float alpha, const float *x, int inc_x, const float *y, int inc_y, float *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(sger, SGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(dger, DGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void cger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int inc_x, const std::complex<float> *y, int inc_y,
          std::complex<float> *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(cgeru, CGERU)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void zger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int inc_x, const std::complex<double> *y, int inc_y,
          std::complex<double> *a, int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(zgeru, ZGERU)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

auto sgetrf(int m, int n, float *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(sgetrf, SGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto dgetrf(int m, int n, double *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(dgetrf, DGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto cgetrf(int m, int n, std::complex<float> *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(cgetrf, CGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto zgetrf(int m, int n, std::complex<double> *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    int info{0};
    FC_GLOBAL(zgetrf, ZGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto sgetri(int n, float *a, int lda, const int *ipiv) -> int {
    LabeledSection0();

    int                info{0};
    int                lwork = n * 64;
    std::vector<float> work(lwork);
    FC_GLOBAL(sgetri, SGETRI)(&n, a, &lda, (int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto dgetri(int n, double *a, int lda, const int *ipiv) -> int {
    LabeledSection0();

    int                 info{0};
    int                 lwork = n * 64;
    std::vector<double> work(lwork);
    FC_GLOBAL(dgetri, DGETRI)(&n, a, &lda, (int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto cgetri(int n, std::complex<float> *a, int lda, const int *ipiv) -> int {
    LabeledSection0();

    int                              info{0};
    int                              lwork = n * 64;
    std::vector<std::complex<float>> work(lwork);
    FC_GLOBAL(cgetri, CGETRI)(&n, a, &lda, (int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto zgetri(int n, std::complex<double> *a, int lda, const int *ipiv) -> int {
    LabeledSection0();

    int                               info{0};
    int                               lwork = n * 64;
    std::vector<std::complex<double>> work(lwork);
    FC_GLOBAL(zgetri, ZGETRI)(&n, a, &lda, (int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto slange(char norm_type, int m, int n, const float *A, int lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(slange, SLANGE)(norm_type, m, n, A, lda, work);
}

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(dlange, DLANGE)(norm_type, m, n, A, lda, work);
}

auto clange(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(clange, CLANGE)(norm_type, m, n, A, lda, work);
}

auto zlange(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(zlange, ZLANGE)(norm_type, m, n, A, lda, work);
}

void slassq(int n, const float *x, int incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(slassq, SLASSQ)(&n, x, &incx, scale, sumsq);
}

void dlassq(int n, const double *x, int incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(dlassq, DLASSQ)(&n, x, &incx, scale, sumsq);
}

void classq(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(classq, CLASSQ)(&n, x, &incx, scale, sumsq);
}

void zlassq(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(zlassq, ZLASSQ)(&n, x, &incx, scale, sumsq);
}

auto dgesdd(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt/*, double *work, int lwork,
            int *iwork */) -> int {
    LabeledSection0();

    // // Query optimal workspace size
    // int    info{0};
    // int    lwork{-1};
    // double work_query;
    // FC_GLOBAL(dgesdd, DGESDD)(&jobz, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu, &work_query, &lwork, nullptr, &info);
    // lwork = static_cast<int>(work_query);

    // // Allocate work array
    // double *work = new double[lwork];

    // // Allocate iwork array
    // int *iwork = new int[8 * std::min(m, n)];

    // // Compute SVD
    // FC_GLOBAL(dgesdd, DGESDD)(&jobz, &n, &m, a, &lda, s, vt, &ldvt, u, &ldu, work, &lwork, iwork, &info);

    // // Free workspace arrays
    // delete[] work;
    // delete[] iwork;

    // return info;

    int nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;
    int ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);
    int nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);

    int lda_t  = std::max(1, m);
    int ldu_t  = std::max(1, nrows_u);
    int ldvt_t = std::max(1, nrows_vt);

    // double *a_t  = nullptr;
    // double *u_t  = nullptr;
    // double *vt_t = nullptr;
    std::vector<double> a_t, u_t, vt_t;

    // Check leading dimensions(s)
    if (lda < n) {
        println_warn("gesdd warning: lda < n, lda = {}, n = {}", lda, n);
        return -5;
    }
    if (ldu < ncols_u) {
        println_warn("gesdd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);
        return -8;
    }
    if (ldvt < n) {
        println_warn("gesdd warning: ldvt < n, ldvt = {}, n = {}", ldvt, n);
        return -10;
    }

    // Query optimial working array(s)
    int    info{0};
    int    lwork{-1};
    double work_query;
    FC_GLOBAL(dgesdd, DGESDD)(&jobz, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, nullptr, &info);

    // Allocate memory for temporary arrays(s)
    a_t.resize(lda_t * std::max(1, n));
    if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {
        u_t.resize(ldu_t * std::max(1, ncols_u));
    }
    if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {
        vt_t.resize(ldvt_t * std::max(1, n));
    }

    // Allocate work array
    std::vector<double> work(lwork);

    // Allocate iwork array
    std::vector<int> iwork(8 * std::min(m, n));

    // Transpose input matrices
    transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);

    // Call lapack routine
    FC_GLOBAL(dgesdd, DGESDD)
    (&jobz, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, iwork.data(), &info);

    // Transpose output matrices
    transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);
    if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {
        transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);
    }
    if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {
        transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);
    }

    return 0;
}

END_EINSUMS_NAMESPACE_CPP(einsums::backend::vendor)