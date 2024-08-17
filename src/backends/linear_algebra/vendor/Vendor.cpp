//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Vendor.hpp"

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Print.hpp"
#include "einsums/Section.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <stdexcept>

#include "Utilities.hpp"

#if defined(EINSUMS_HAVE_MKL)
typedef void (*XerblaEntry)(const char *Name, const int *Num, const int Len);
extern "C" {
XerblaEntry mkl_set_xerbla(XerblaEntry xerbla);
}
#endif

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

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::backend::linear_algebra::vendor)

EINSUMS_DISABLE_WARNING_PUSH
EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
extern "C" {

extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, blas_int *, blas_int *, blas_int *, float *, const float *, blas_int *, const float *,
                                    blas_int *, float *, float *, blas_int *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, blas_int *, blas_int *, blas_int *, double *, const double *, blas_int *,
                                    const double *, blas_int *, double *, double *, blas_int *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, blas_int *, blas_int *, blas_int *, std::complex<float> *, const std::complex<float> *,
                                    blas_int *, const std::complex<float> *, blas_int *, std::complex<float> *, std::complex<float> *,
                                    blas_int *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, blas_int *, blas_int *, blas_int *, std::complex<double> *,
                                    const std::complex<double> *, blas_int *, const std::complex<double> *, blas_int *,
                                    std::complex<double> *, std::complex<double> *, blas_int *);

extern void FC_GLOBAL(sgemv, SGEMV)(char *, blas_int *, blas_int *, float *, const float *, blas_int *, const float *, blas_int *, float *,
                                    float *, blas_int *);
extern void FC_GLOBAL(dgemv, DGEMV)(char *, blas_int *, blas_int *, double *, const double *, blas_int *, const double *, blas_int *,
                                    double *, double *, blas_int *);
extern void FC_GLOBAL(cgemv, CGEMV)(char *, blas_int *, blas_int *, std::complex<float> *, const std::complex<float> *, blas_int *,
                                    const std::complex<float> *, blas_int *, std::complex<float> *, std::complex<float> *, blas_int *);
extern void FC_GLOBAL(zgemv, ZGEMV)(char *, blas_int *, blas_int *, std::complex<double> *, const std::complex<double> *, blas_int *,
                                    const std::complex<double> *, blas_int *, std::complex<double> *, std::complex<double> *, blas_int *);

extern void FC_GLOBAL(cheev, CHEEV)(char *job, char *uplo, blas_int *n, std::complex<float> *a, blas_int *lda, float *w,
                                    std::complex<float> *work, blas_int *lwork, float *rwork, blas_int *info);
extern void FC_GLOBAL(zheev, ZHEEV)(char *job, char *uplo, blas_int *n, std::complex<double> *a, blas_int *lda, double *w,
                                    std::complex<double> *work, blas_int *lwork, double *rwork, blas_int *info);

extern void FC_GLOBAL(ssyev, SSYEV)(char *, char *, blas_int *, float *, blas_int *, float *, float *, blas_int *, blas_int *);
extern void FC_GLOBAL(dsyev, DSYEV)(char *, char *, blas_int *, double *, blas_int *, double *, double *, blas_int *, blas_int *);

extern void FC_GLOBAL(sgeev, SGEEV)(char *, char *, blas_int *, float *, blas_int *, float *, float *, float *, blas_int *, float *,
                                    blas_int *, float *, blas_int *, blas_int *);
extern void FC_GLOBAL(dgeev, DGEEV)(char *, char *, blas_int *, double *, blas_int *, double *, double *, double *, blas_int *, double *,
                                    blas_int *, double *, blas_int *, blas_int *);
extern void FC_GLOBAL(cgeev, CGEEV)(char *, char *, blas_int *, std::complex<float> *, blas_int *, std::complex<float> *,
                                    std::complex<float> *, blas_int *, std::complex<float> *, blas_int *, std::complex<float> *, blas_int *,
                                    float *, blas_int *);
extern void FC_GLOBAL(zgeev, ZGEEV)(char *, char *, blas_int *, std::complex<double> *, blas_int *, std::complex<double> *,
                                    std::complex<double> *, blas_int *, std::complex<double> *, blas_int *, std::complex<double> *,
                                    blas_int *, double *, blas_int *);

extern void FC_GLOBAL(sgesv, SGESV)(blas_int *, blas_int *, float *, blas_int *, blas_int *, float *, blas_int *, blas_int *);
extern void FC_GLOBAL(dgesv, DGESV)(blas_int *, blas_int *, double *, blas_int *, blas_int *, double *, blas_int *, blas_int *);
extern void FC_GLOBAL(cgesv, CGESV)(blas_int *, blas_int *, std::complex<float> *, blas_int *, blas_int *, std::complex<float> *,
                                    blas_int *, blas_int *);
extern void FC_GLOBAL(zgesv, ZGESV)(blas_int *, blas_int *, std::complex<double> *, blas_int *, blas_int *, std::complex<double> *,
                                    blas_int *, blas_int *);

extern void FC_GLOBAL(sscal, SSCAL)(blas_int *, float *, float *, blas_int *);
extern void FC_GLOBAL(dscal, DSCAL)(blas_int *, double *, double *, blas_int *);
extern void FC_GLOBAL(cscal, CSCAL)(blas_int *, std::complex<float> *, std::complex<float> *, blas_int *);
extern void FC_GLOBAL(zscal, ZSCAL)(blas_int *, std::complex<double> *, std::complex<double> *, blas_int *);
extern void FC_GLOBAL(csscal, CSSCAL)(blas_int *, float *, std::complex<float> *, blas_int *);
extern void FC_GLOBAL(zdscal, ZDSCAL)(blas_int *, double *, std::complex<double> *, blas_int *);

extern float  FC_GLOBAL(sdot, SDOT)(blas_int *, const float *, blas_int *, const float *, blas_int *);
extern double FC_GLOBAL(ddot, DDOT)(blas_int *, const double *, blas_int *, const double *, blas_int *);
extern std::complex<float>  FC_GLOBAL(cdotc, CDOTC)(blas_int *, const std::complex<float> *, blas_int *, const std::complex<float> *, blas_int *);
extern std::complex<double> FC_GLOBAL(zdotc, ZDOTC)(blas_int *, const std::complex<double> *, blas_int *, const std::complex<double> *, blas_int *);
// MKL seems to have a different function signature than openblas.
// extern std::complex<float>  FC_GLOBAL(cdotu, CDOTU)(blas_int *, const std::complex<float> *, blas_int *, const std::complex<float> *,
//                                                    blas_int *);
// extern std::complex<double> FC_GLOBAL(zdotu, ZDOTU)(blas_int *, const std::complex<double> *, blas_int *, const std::complex<double> *,
//                                                     blas_int *);

extern void FC_GLOBAL(saxpy, SAXPY)(blas_int *, float *, const float *, blas_int *, float *, blas_int *);
extern void FC_GLOBAL(daxpy, DAXPY)(blas_int *, double *, const double *, blas_int *, double *, blas_int *);
extern void FC_GLOBAL(caxpy, CAXPY)(blas_int *, std::complex<float> *, const std::complex<float> *, blas_int *, std::complex<float> *,
                                    blas_int *);
extern void FC_GLOBAL(zaxpy, ZAXPY)(blas_int *, std::complex<double> *, const std::complex<double> *, blas_int *, std::complex<double> *,
                                    blas_int *);

extern void FC_GLOBAL(sger, DGER)(blas_int *, blas_int *, float *, const float *, blas_int *, const float *, blas_int *, float *,
                                  blas_int *);
extern void FC_GLOBAL(dger, DGER)(blas_int *, blas_int *, double *, const double *, blas_int *, const double *, blas_int *, double *,
                                  blas_int *);
extern void FC_GLOBAL(cgeru, CGERU)(blas_int *, blas_int *, std::complex<float> *, const std::complex<float> *, blas_int *,
                                    const std::complex<float> *, blas_int *, std::complex<float> *, blas_int *);
extern void FC_GLOBAL(zgeru, ZGERU)(blas_int *, blas_int *, std::complex<double> *, const std::complex<double> *, blas_int *,
                                    const std::complex<double> *, blas_int *, std::complex<double> *, blas_int *);

extern void FC_GLOBAL(sgetrf, SGETRF)(blas_int *, blas_int *, float *, blas_int *, blas_int *, blas_int *);
extern void FC_GLOBAL(dgetrf, DGETRF)(blas_int *, blas_int *, double *, blas_int *, blas_int *, blas_int *);
extern void FC_GLOBAL(cgetrf, CGETRF)(blas_int *, blas_int *, std::complex<float> *, blas_int *, blas_int *, blas_int *);
extern void FC_GLOBAL(zgetrf, ZGETRF)(blas_int *, blas_int *, std::complex<double> *, blas_int *, blas_int *, blas_int *);

extern void FC_GLOBAL(sgetri, SGETRI)(blas_int *, float *, blas_int *, blas_int *, float *, blas_int *, blas_int *);
extern void FC_GLOBAL(dgetri, DGETRI)(blas_int *, double *, blas_int *, blas_int *, double *, blas_int *, blas_int *);
extern void FC_GLOBAL(cgetri, CGETRI)(blas_int *, std::complex<float> *, blas_int *, blas_int *, std::complex<float> *, blas_int *,
                                      blas_int *);
extern void FC_GLOBAL(zgetri, ZGETRI)(blas_int *, std::complex<double> *, blas_int *, blas_int *, std::complex<double> *, blas_int *,
                                      blas_int *);

// According to my Xcode 15.3 macOS 14.4 SDK:
// The Accelerate clapack.h header does use double for Xlange's. However, that interface is deprecated according to the headers.
// The "new lapack" Accelerate header does use the following return types. For now, we're going to leave it as is.
// If it becomes an issue then we'll need to do something about it.
extern float  FC_GLOBAL(slange, SLANGE)(const char *, blas_int *, blas_int *, const float *, blas_int *, float *);
extern double FC_GLOBAL(dlange, DLANGE)(const char *, blas_int *, blas_int *, const double *, blas_int *, double *);
extern float  FC_GLOBAL(clange, CLANGE)(const char *, blas_int *, blas_int *, const std::complex<float> *, blas_int *, float *);
extern double FC_GLOBAL(zlange, ZLANGE)(const char *, blas_int *, blas_int *, const std::complex<double> *, blas_int *, double *);

extern void FC_GLOBAL(slassq, SLASSQ)(blas_int *n, const float *x, blas_int *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(dlassq, DLASSQ)(blas_int *n, const double *x, blas_int *incx, double *scale, double *sumsq);
extern void FC_GLOBAL(classq, CLASSQ)(blas_int *n, const std::complex<float> *x, blas_int *incx, float *scale, float *sumsq);
extern void FC_GLOBAL(zlassq, ZLASSQ)(blas_int *n, const std::complex<double> *x, blas_int *incx, double *scale, double *sumsq);

extern void FC_GLOBAL(dgesvd, DGESVD)(char *, char *, blas_int *, blas_int *, double *, blas_int *, double *, double *, blas_int *,
                                      double *, blas_int *, double *, blas_int *, blas_int *);
extern void FC_GLOBAL(sgesvd, SGESVD)(char *, char *, blas_int *, blas_int *, float *, blas_int *, float *, float *, blas_int *, float *,
                                      blas_int *, float *, blas_int *, blas_int *);

extern void FC_GLOBAL(dgesdd, DGESDD)(char *, blas_int *, blas_int *, double *, blas_int *, double *, double *, blas_int *, double *,
                                      blas_int *, double *, blas_int *, blas_int *, blas_int *);
extern void FC_GLOBAL(sgesdd, SGESDD)(char *, blas_int *, blas_int *, float *, blas_int *, float *, float *, blas_int *, float *,
                                      blas_int *, float *, blas_int *, blas_int *, blas_int *);
extern void FC_GLOBAL(zgesdd, ZGESDD)(char *, blas_int *, blas_int *, std::complex<double> *, blas_int *, double *, std::complex<double> *,
                                      blas_int *, std::complex<double> *, blas_int *, std::complex<double> *, blas_int *, double *,
                                      blas_int *, blas_int *);
extern void FC_GLOBAL(cgesdd, CGESDD)(char *, blas_int *, blas_int *, std::complex<float> *, blas_int *, float *, std::complex<float> *,
                                      blas_int *, std::complex<float> *, blas_int *, std::complex<float> *, blas_int *, float *, blas_int *,
                                      blas_int *);

extern void FC_GLOBAL(dgees, DGEES)(char *, char *, blas_int (*)(double *, double *), blas_int *, double *, blas_int *, blas_int *,
                                    double *, double *, double *, blas_int *, double *, blas_int *, blas_int *, blas_int *);
extern void FC_GLOBAL(sgees, SGEES)(char *, char *, blas_int (*)(float *, float *), blas_int *, float *, blas_int *, blas_int *, float *,
                                    float *, float *, blas_int *, float *, blas_int *, blas_int *, blas_int *);

extern void FC_GLOBAL(dtrsyl, DTRSYL)(char *, char *, blas_int *, blas_int *, blas_int *, const double *, blas_int *, const double *,
                                      blas_int *, double *, blas_int *, double *, blas_int *);
extern void FC_GLOBAL(strsyl, STRSYL)(char *, char *, blas_int *, blas_int *, blas_int *, const float *, blas_int *, const float *,
                                      blas_int *, float *, blas_int *, float *, blas_int *);

extern void FC_GLOBAL(sorgqr, SORGQR)(blas_int *, blas_int *, blas_int *, float *, blas_int *, const float *, const float *, blas_int *,
                                      blas_int *);
extern void FC_GLOBAL(dorgqr, DORGQR)(blas_int *, blas_int *, blas_int *, double *, blas_int *, const double *, const double *, blas_int *,
                                      blas_int *);
extern void FC_GLOBAL(cungqr, CUNGQR)(blas_int *, blas_int *, blas_int *, std::complex<float> *, blas_int *, const std::complex<float> *,
                                      const std::complex<float> *, blas_int *, blas_int *);
extern void FC_GLOBAL(zungqr, ZUNGQR)(blas_int *, blas_int *, blas_int *, std::complex<double> *, blas_int *, const std::complex<double> *,
                                      const std::complex<double> *, blas_int *, blas_int *);

extern void FC_GLOBAL(sgeqrf, SGEQRF)(blas_int *, blas_int *, float *, blas_int *, float *, float *, blas_int *, blas_int *);
extern void FC_GLOBAL(dgeqrf, DGEQRF)(blas_int *, blas_int *, double *, blas_int *, double *, double *, blas_int *, blas_int *);
extern void FC_GLOBAL(cgeqrf, CGEQRF)(blas_int *, blas_int *, std::complex<float> *, blas_int *, std::complex<float> *,
                                      std::complex<float> *, blas_int *, blas_int *);
extern void FC_GLOBAL(zgeqrf, ZGEQRF)(blas_int *, blas_int *, std::complex<double> *, blas_int *, std::complex<double> *,
                                      std::complex<double> *, blas_int *, blas_int *);
} // extern "C"
EINSUMS_DISABLE_WARNING_POP

namespace {
extern "C" void xerbla(const char *srname, const int *info, const int len) {
    if (*info == 1001) {
        println_abort("BLAS/LAPACK: Incompatible optional parameters on entry to {}", srname);
    } else if (*info == 1000 || *info == 1089) {
        println_abort("BLAS/LAPACK: Insufficient workspace available in function {}.", srname);
    } else if (*info < 0) {
        println_abort("BLAS/LAPACK: Condition {} detected in function {}}.", -(*info), srname);
    } else {
        println_abort("BLAS/LAPACK: The value of parameter {} is invalid in function call to {}.", *info, srname);
    }
}

} // namespace

void initialize() {
#if defined(EINSUMS_HAVE_MKL)
    mkl_set_xerbla(&xerbla);
#endif
}

void finalize() {
}

void sgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, float alpha, const float *a, blas_int lda, const float *b,
           blas_int ldb, float beta, float *c, blas_int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(sgemm, SGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, double alpha, const double *a, blas_int lda, const double *b,
           blas_int ldb, double beta, double *c, blas_int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(dgemm, DGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void cgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<float> alpha, const std::complex<float> *a,
           blas_int lda, const std::complex<float> *b, blas_int ldb, std::complex<float> beta, std::complex<float> *c, blas_int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(cgemm, CGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void zgemm(char transa, char transb, blas_int m, blas_int n, blas_int k, std::complex<double> alpha, const std::complex<double> *a,
           blas_int lda, const std::complex<double> *b, blas_int ldb, std::complex<double> beta, std::complex<double> *c, blas_int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(zgemm, ZGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void sgemv(char transa, blas_int m, blas_int n, float alpha, const float *a, blas_int lda, const float *x, blas_int incx, float beta,
           float *y, blas_int incy) {
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

void dgemv(char transa, blas_int m, blas_int n, double alpha, const double *a, blas_int lda, const double *x, blas_int incx, double beta,
           double *y, blas_int incy) {
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

void cgemv(char transa, blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *a, blas_int lda,
           const std::complex<float> *x, blas_int incx, std::complex<float> beta, std::complex<float> *y, blas_int incy) {
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

void zgemv(char transa, blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *a, blas_int lda,
           const std::complex<double> *x, blas_int incx, std::complex<double> beta, std::complex<double> *y, blas_int incy) {
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

auto ssyev(char job, char uplo, blas_int n, float *a, blas_int lda, float *w, float *work, blas_int lwork) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(ssyev, SSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dsyev(char job, char uplo, blas_int n, double *a, blas_int lda, double *w, double *work, blas_int lwork) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(dsyev, DSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto cheev(char job, char uplo, blas_int n, std::complex<float> *a, blas_int lda, float *w, std::complex<float> *work, blas_int lwork,
           float *rwork) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(cheev, CHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}
auto zheev(char job, char uplo, blas_int n, std::complex<double> *a, blas_int lda, double *w, std::complex<double> *work, blas_int lwork,
           double *rwork) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(zheev, ZHEEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    return info;
}

auto sgesv(blas_int n, blas_int nrhs, float *a, blas_int lda, blas_int *ipiv, float *b, blas_int ldb) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(sgesv, SGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto dgesv(blas_int n, blas_int nrhs, double *a, blas_int lda, blas_int *ipiv, double *b, blas_int ldb) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(dgesv, DGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto cgesv(blas_int n, blas_int nrhs, std::complex<float> *a, blas_int lda, blas_int *ipiv, std::complex<float> *b,
           blas_int ldb) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(cgesv, CGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

auto zgesv(blas_int n, blas_int nrhs, std::complex<double> *a, blas_int lda, blas_int *ipiv, std::complex<double> *b,
           blas_int ldb) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(zgesv, ZGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

void sscal(blas_int n, float alpha, float *vec, blas_int inc) {
    LabeledSection0();

    FC_GLOBAL(sscal, SSCAL)(&n, &alpha, vec, &inc);
}

void dscal(blas_int n, double alpha, double *vec, blas_int inc) {
    LabeledSection0();

    FC_GLOBAL(dscal, DSCAL)(&n, &alpha, vec, &inc);
}

void cscal(blas_int n, std::complex<float> alpha, std::complex<float> *vec, blas_int inc) {
    LabeledSection0();

    FC_GLOBAL(cscal, CSCAL)(&n, &alpha, vec, &inc);
}

void zscal(blas_int n, std::complex<double> alpha, std::complex<double> *vec, blas_int inc) {
    LabeledSection0();

    FC_GLOBAL(zscal, ZSCAL)(&n, &alpha, vec, &inc);
}

void csscal(blas_int n, float alpha, std::complex<float> *vec, blas_int inc) {
    LabeledSection0();

    FC_GLOBAL(csscal, CSSCAL)(&n, &alpha, vec, &inc);
}

void zdscal(blas_int n, double alpha, std::complex<double> *vec, blas_int inc) {
    LabeledSection0();

    FC_GLOBAL(zdscal, ZDSCAL)(&n, &alpha, vec, &inc);
}

auto sdot(blas_int n, const float *x, blas_int incx, const float *y, blas_int incy) -> float {
    LabeledSection0();

    return FC_GLOBAL(sdot, SDOT)(&n, x, &incx, y, &incy);
}

auto ddot(blas_int n, const double *x, blas_int incx, const double *y, blas_int incy) -> double {
    LabeledSection0();

    return FC_GLOBAL(ddot, DDOT)(&n, x, &incx, y, &incy);
}

// We implement the cdotu as the default for cdot.
auto cdot(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y, blas_int incy) -> std::complex<float> {
    LabeledSection0();

    // Since MKL does not conform to the netlib standard, we need to use the following code.
    std::complex<float> result{0.0F, 0.0F};
    for (blas_int i = 0; i < n; ++i) {
        result += x[static_cast<ptrdiff_t>(i * incx)] * y[static_cast<ptrdiff_t>(i * incy)];
    }
    return result;
}

// We implement the zdotu as the default for cdot.
auto zdot(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y, blas_int incy) -> std::complex<double> {
    LabeledSection0();

    // Since MKL does not conform to the netlib standard, we need to use the following code.
    std::complex<double> result{0.0, 0.0};
    for (blas_int i = 0; i < n; ++i) {
        result += x[static_cast<ptrdiff_t>(i * incx)] * y[static_cast<ptrdiff_t>(i * incy)];
    }
    return result;
}

auto cdotc(blas_int n, const std::complex<float> *x, blas_int incx, const std::complex<float> *y, blas_int incy) -> std::complex<float> {
    LabeledSection0();

    return FC_GLOBAL(cdotc, CDOTC)(&n, x, &incx, y, &incy);
}

auto zdotc(blas_int n, const std::complex<double> *x, blas_int incx, const std::complex<double> *y, blas_int incy) -> std::complex<double> {
    LabeledSection0();

    return FC_GLOBAL(zdotc, ZDOTC)(&n, x, &incx, y, &incy);
}

void saxpy(blas_int n, float alpha_x, const float *x, blas_int inc_x, float *y, blas_int inc_y) {
    LabeledSection0();

    FC_GLOBAL(saxpy, SAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void daxpy(blas_int n, double alpha_x, const double *x, blas_int inc_x, double *y, blas_int inc_y) {
    LabeledSection0();

    FC_GLOBAL(daxpy, DAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void caxpy(blas_int n, std::complex<float> alpha_x, const std::complex<float> *x, blas_int inc_x, std::complex<float> *y, blas_int inc_y) {
    LabeledSection0();

    FC_GLOBAL(caxpy, CAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void zaxpy(blas_int n, std::complex<double> alpha_x, const std::complex<double> *x, blas_int inc_x, std::complex<double> *y,
           blas_int inc_y) {
    LabeledSection0();

    FC_GLOBAL(zaxpy, ZAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void saxpby(const blas_int n, const float a, const float *x, const blas_int incx, const float b, float *y, const blas_int incy) {
    LabeledSection0();
    sscal(n, b, y, incy);
    saxpy(n, a, x, incx, y, incy);
}

void daxpby(const blas_int n, const double a, const double *x, const blas_int incx, const double b, double *y, const blas_int incy) {
    LabeledSection0();
    dscal(n, b, y, incy);
    daxpy(n, a, x, incx, y, incy);
}

void caxpby(const blas_int n, const std::complex<float> a, const std::complex<float> *x, const blas_int incx, const std::complex<float> b,
            std::complex<float> *y, const blas_int incy) {
    LabeledSection0();
    cscal(n, b, y, incy);
    caxpy(n, a, x, incx, y, incy);
}

void zaxpby(const blas_int n, const std::complex<double> a, const std::complex<double> *x, const blas_int incx,
            const std::complex<double> b, std::complex<double> *y, const blas_int incy) {
    LabeledSection0();
    zscal(n, b, y, incy);
    zaxpy(n, a, x, incx, y, incy);
}

namespace {
void ger_parameter_check(blas_int m, blas_int n, blas_int inc_x, blas_int inc_y, blas_int lda) {
    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: m ({}) is less than zero.", m));
    }
    if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: n ({}) is less than zero.", n));
    }
    if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: inc_x ({}) is zero.", inc_x));
    }
    if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: inc_y ({}) is zero.", inc_y));
    }
    if (lda < std::max(blas_int{1}, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::ger: lda ({}) is less than max(1, n ({})).", lda, n));
    }
}
} // namespace

void sger(blas_int m, blas_int n, float alpha, const float *x, blas_int inc_x, const float *y, blas_int inc_y, float *a, blas_int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(sger, SGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void dger(blas_int m, blas_int n, double alpha, const double *x, blas_int inc_x, const double *y, blas_int inc_y, double *a, blas_int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(dger, DGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void cger(blas_int m, blas_int n, std::complex<float> alpha, const std::complex<float> *x, blas_int inc_x, const std::complex<float> *y,
          blas_int inc_y, std::complex<float> *a, blas_int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(cgeru, CGERU)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

void zger(blas_int m, blas_int n, std::complex<double> alpha, const std::complex<double> *x, blas_int inc_x, const std::complex<double> *y,
          blas_int inc_y, std::complex<double> *a, blas_int lda) {
    LabeledSection0();

    ger_parameter_check(m, n, inc_x, inc_y, lda);
    FC_GLOBAL(zgeru, ZGERU)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

auto sgetrf(blas_int m, blas_int n, float *a, blas_int lda, blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(sgetrf, SGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto dgetrf(blas_int m, blas_int n, double *a, blas_int lda, blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(dgetrf, DGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto cgetrf(blas_int m, blas_int n, std::complex<float> *a, blas_int lda, blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(cgetrf, CGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto zgetrf(blas_int m, blas_int n, std::complex<double> *a, blas_int lda, blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int info{0};
    FC_GLOBAL(zgetrf, ZGETRF)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

auto sgetri(blas_int n, float *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int           info{0};
    blas_int           lwork = n * 64;
    std::vector<float> work(lwork);
    FC_GLOBAL(sgetri, SGETRI)(&n, a, &lda, (blas_int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto dgetri(blas_int n, double *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int            info{0};
    blas_int            lwork = n * 64;
    std::vector<double> work(lwork);
    FC_GLOBAL(dgetri, DGETRI)(&n, a, &lda, (blas_int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto cgetri(blas_int n, std::complex<float> *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int                         info{0};
    blas_int                         lwork = n * 64;
    std::vector<std::complex<float>> work(lwork);
    FC_GLOBAL(cgetri, CGETRI)(&n, a, &lda, (blas_int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto zgetri(blas_int n, std::complex<double> *a, blas_int lda, const blas_int *ipiv) -> blas_int {
    LabeledSection0();

    blas_int                          info{0};
    blas_int                          lwork = n * 64;
    std::vector<std::complex<double>> work(lwork);
    FC_GLOBAL(zgetri, ZGETRI)(&n, a, &lda, (blas_int *)ipiv, work.data(), &lwork, &info);
    return info;
}

auto slange(char norm_type, blas_int m, blas_int n, const float *A, blas_int lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(slange, SLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto dlange(char norm_type, blas_int m, blas_int n, const double *A, blas_int lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(dlange, DLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto clange(char norm_type, blas_int m, blas_int n, const std::complex<float> *A, blas_int lda, float *work) -> float {
    LabeledSection0();

    return FC_GLOBAL(clange, CLANGE)(&norm_type, &m, &n, A, &lda, work);
}

auto zlange(char norm_type, blas_int m, blas_int n, const std::complex<double> *A, blas_int lda, double *work) -> double {
    LabeledSection0();

    return FC_GLOBAL(zlange, ZLANGE)(&norm_type, &m, &n, A, &lda, work);
}

void slassq(blas_int n, const float *x, blas_int incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(slassq, SLASSQ)(&n, x, &incx, scale, sumsq);
}

void dlassq(blas_int n, const double *x, blas_int incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(dlassq, DLASSQ)(&n, x, &incx, scale, sumsq);
}

void classq(blas_int n, const std::complex<float> *x, blas_int incx, float *scale, float *sumsq) {
    LabeledSection0();

    FC_GLOBAL(classq, CLASSQ)(&n, x, &incx, scale, sumsq);
}

void zlassq(blas_int n, const std::complex<double> *x, blas_int incx, double *scale, double *sumsq) {
    LabeledSection0();

    FC_GLOBAL(zlassq, ZLASSQ)(&n, x, &incx, scale, sumsq);
}

#define GESDD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesdd(char jobz, blas_int m, blas_int n, Type *a, blas_int lda, Type *s, Type *u, blas_int ldu, Type *vt,               \
                         blas_int ldvt)                                                                                                    \
        ->blas_int {                                                                                                                       \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;                                 \
        blas_int ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);               \
        blas_int nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);              \
                                                                                                                                           \
        blas_int          lda_t  = std::max(blas_int{1}, m);                                                                               \
        blas_int          ldu_t  = std::max(blas_int{1}, nrows_u);                                                                         \
        blas_int          ldvt_t = std::max(blas_int{1}, nrows_vt);                                                                        \
        std::vector<Type> a_t, u_t, vt_t;                                                                                                  \
                                                                                                                                           \
        /* Check leading dimensions(s) */                                                                                                  \
        if (lda < n) {                                                                                                                     \
            println_warn("gesdd warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            println_warn("gesdd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                            \
            return -8;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvt < n) {                                                                                                                    \
            println_warn("gesdd warning: ldvt < n, ldvt = {}, n = {}", ldvt, n);                                                           \
            return -10;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) */                                                                                               \
        blas_int info{0};                                                                                                                  \
        blas_int lwork{-1};                                                                                                                \
        Type     work_query;                                                                                                               \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, nullptr, &info);                                        \
        lwork = (int)work_query;                                                                                                           \
                                                                                                                                           \
        /* Allocate memory for temporary arrays(s) */                                                                                      \
        a_t.resize(lda_t *std::max(blas_int{1}, n));                                                                                       \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            u_t.resize(ldu_t *std::max(blas_int{1}, ncols_u));                                                                             \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            vt_t.resize(ldvt_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate work array */                                                                                                          \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate iwork array */                                                                                                         \
        std::vector<blas_int> iwork(8 * std::min(m, n));                                                                                   \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, iwork.data(), &info);        \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);                                                           \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);                                                            \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

#define GESDD_complex(Type, lc, UC)                                                                                                        \
    auto lc##gesdd(char jobz, blas_int m, blas_int n, std::complex<Type> *a, blas_int lda, Type *s, std::complex<Type> *u, blas_int ldu,   \
                   std::complex<Type> *vt, blas_int ldvt)                                                                                  \
        ->blas_int {                                                                                                                       \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;                                 \
        blas_int ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);               \
        blas_int nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);              \
                                                                                                                                           \
        blas_int                        lda_t  = std::max(blas_int{1}, m);                                                                 \
        blas_int                        ldu_t  = std::max(blas_int{1}, nrows_u);                                                           \
        blas_int                        ldvt_t = std::max(blas_int{1}, nrows_vt);                                                          \
        blas_int                        info{0};                                                                                           \
        blas_int                        lwork{-1};                                                                                         \
        size_t                          lrwork;                                                                                            \
        std::complex<Type>              work_query;                                                                                        \
        std::vector<std::complex<Type>> a_t, u_t, vt_t;                                                                                    \
        std::vector<Type>               rwork;                                                                                             \
        std::vector<std::complex<Type>> work;                                                                                              \
        std::vector<blas_int>           iwork;                                                                                             \
                                                                                                                                           \
        /* Check leading dimensions(s) */                                                                                                  \
        if (lda < n) {                                                                                                                     \
            println_warn("gesdd warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            println_warn("gesdd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                            \
            return -8;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvt < n) {                                                                                                                    \
            println_warn("gesdd warning: ldvt < n, ldvt = {}, n = {}", ldvt, n);                                                           \
            return -10;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        if (lsame(jobz, 'n')) {                                                                                                            \
            lrwork = std::max(blas_int{1}, 7 * std::min(m, n));                                                                            \
        } else {                                                                                                                           \
            lrwork = (size_t)std::max(blas_int{1},                                                                                         \
                                      std::min(m, n) * std::max(5 * std::min(m, n) + 7, 2 * std::max(m, n) + 2 * std::min(m, n) + 1));     \
        }                                                                                                                                  \
                                                                                                                                           \
        iwork.resize(std::max(blas_int{1}, 8 * std::min(m, n)));                                                                           \
        rwork.resize(lrwork);                                                                                                              \
                                                                                                                                           \
        /* Query optimal working array(s) */                                                                                               \
        FC_GLOBAL(lc##gesdd, UC##GESDD)                                                                                                    \
        (&jobz, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, rwork.data(), iwork.data(), &info);                     \
        lwork = (int)(work_query.real());                                                                                                  \
                                                                                                                                           \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        /* Allocate memory for temporary arrays(s) */                                                                                      \
        a_t.resize(lda_t *std::max(blas_int{1}, n));                                                                                       \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            u_t.resize(ldu_t *std::max(blas_int{1}, ncols_u));                                                                             \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            vt_t.resize(ldvt_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lc##gesdd, UC##GESDD)                                                                                                    \
        (&jobz, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, rwork.data(), iwork.data(),  \
         &info);                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            println_warn("gesdd lapack routine failed. info {}", info);                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);                                                           \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);                                                            \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESDD(double, d, D);
GESDD(float, s, S);
GESDD_complex(float, c, C);
GESDD_complex(double, z, Z);

#define GESVD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesvd(char jobu, char jobvt, blas_int m, blas_int n, Type *a, blas_int lda, Type *s, Type *u, blas_int ldu, Type *vt,   \
                         blas_int ldvt, Type *superb)                                                                                      \
        ->blas_int {                                                                                                                       \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int info  = 0;                                                                                                                \
        blas_int lwork = -1;                                                                                                               \
                                                                                                                                           \
        Type     work_query;                                                                                                               \
        blas_int i;                                                                                                                        \
                                                                                                                                           \
        blas_int nrows_u  = (lsame(jobu, 'a') || lsame(jobu, 's')) ? m : 1;                                                                \
        blas_int ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std::min(m, n) : 1);                                                \
        blas_int nrows_vt = lsame(jobvt, 'a') ? n : (lsame(jobvt, 's') ? std::min(m, n) : 1);                                              \
        blas_int ncols_vt = (lsame(jobvt, 'a') || lsame(jobvt, 's')) ? n : 1;                                                              \
                                                                                                                                           \
        blas_int lda_t  = std::max(blas_int{1}, m);                                                                                        \
        blas_int ldu_t  = std::max(blas_int{1}, nrows_u);                                                                                  \
        blas_int ldvt_t = std::max(blas_int{1}, nrows_vt);                                                                                 \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("gesvd warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -6;                                                                                                                     \
        }                                                                                                                                  \
        if (ldu < ncols_u) {                                                                                                               \
            println_warn("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);                                            \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvt < ncols_vt) {                                                                                                             \
            println_warn("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);                                      \
            return -11;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array(s) size */                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, &info);                                         \
        if (info != 0)                                                                                                                     \
            println_abort("gesvd work array size query failed. info {}", info);                                                            \
                                                                                                                                           \
        lwork = (blas_int)work_query;                                                                                                      \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, n));                                                                            \
        std::vector<Type> u_t, vt_t;                                                                                                       \
        if (lsame(jobu, 'a') || lsame(jobu, 's')) {                                                                                        \
            u_t.resize(ldu_t *std::max(blas_int{1}, ncols_u));                                                                             \
        }                                                                                                                                  \
        if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {                                                                                      \
            vt_t.resize(ldvt_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, &info);              \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            println_abort("gesvd lapack routine failed. info {}", info);                                                                   \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobu, 'a') || lsame(jobu, 's')) {                                                                                        \
            transpose<OrderMajor::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);                                                           \
        }                                                                                                                                  \
        if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {                                                                                      \
            transpose<OrderMajor::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);                                                            \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Backup significant data from working arrays into superb */                                                                      \
        for (i = 0; i < std::min(m, n) - 1; i++) {                                                                                         \
            superb[i] = work[i + 1];                                                                                                       \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GESVD(double, d, D);
GESVD(float, s, S);

#define GEES(Type, lc, UC)                                                                                                                 \
    auto lc##gees(char jobvs, blas_int n, Type *a, blas_int lda, blas_int *sdim, Type *wr, Type *wi, Type *vs, blas_int ldvs)->blas_int {  \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int  info  = 0;                                                                                                               \
        blas_int  lwork = -1;                                                                                                              \
        blas_int *bwork = nullptr;                                                                                                         \
                                                                                                                                           \
        Type work_query;                                                                                                                   \
                                                                                                                                           \
        blas_int lda_t  = std::max(blas_int{1}, n);                                                                                        \
        blas_int ldvs_t = std::max(blas_int{1}, n);                                                                                        \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("gees warning: lda < n, lda = {}, n = {}", lda, n);                                                               \
            return -4;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvs < n) {                                                                                                                    \
            println_warn("gees warning: ldvs < n, ldvs = {}, n = {}", ldvs, n);                                                            \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        char sort = 'N';                                                                                                                   \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a, &lda_t, sdim, wr, wi, vs, &ldvs_t, &work_query, &lwork, bwork, &info);                             \
                                                                                                                                           \
        lwork = (blas_int)work_query;                                                                                                      \
        /* Allocate memory for work array */                                                                                               \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, n));                                                                            \
        std::vector<Type> vs_t;                                                                                                            \
        if (lsame(jobvs, 'v')) {                                                                                                           \
            vs_t.resize(ldvs_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(n, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##gees, UC##GEES)                                                                                                      \
        (&jobvs, &sort, nullptr, &n, a_t.data(), &lda_t, sdim, wr, wi, vs_t.data(), &ldvs_t, work.data(), &lwork, bwork, &info);           \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(n, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobvs, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vs_t, ldvs_t, vs, ldvs);                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GEES(double, d, D);
GEES(float, s, S);

#define TRSYL(Type, lc, uc)                                                                                                                \
    auto lc##trsyl(char trana, char tranb, blas_int isgn, blas_int m, blas_int n, const Type *a, blas_int lda, const Type *b,              \
                   blas_int ldb, Type *c, blas_int ldc, Type *scale)                                                                       \
        ->blas_int {                                                                                                                       \
        blas_int info  = 0;                                                                                                                \
        blas_int lda_t = std::max(blas_int{1}, m);                                                                                         \
        blas_int ldb_t = std::max(blas_int{1}, n);                                                                                         \
        blas_int ldc_t = std::max(blas_int{1}, m);                                                                                         \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < m) {                                                                                                                     \
            println_warn("trsyl warning: lda < m, lda = {}, m = {}", lda, m);                                                              \
            return -7;                                                                                                                     \
        }                                                                                                                                  \
        if (ldb < n) {                                                                                                                     \
            println_warn("trsyl warning: ldb < n, ldb = {}, n = {}", ldb, n);                                                              \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
        if (ldc < n) {                                                                                                                     \
            println_warn("trsyl warning: ldc < n, ldc = {}, n = {}", ldc, n);                                                              \
            return -11;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, m));                                                                            \
        std::vector<Type> b_t(ldb_t *std::max(blas_int{1}, n));                                                                            \
        std::vector<Type> c_t(ldc_t *std::max(blas_int{1}, n));                                                                            \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, m, a, lda, a_t, lda_t);                                                                              \
        transpose<OrderMajor::Row>(n, n, b, ldb, b_t, ldb_t);                                                                              \
        transpose<OrderMajor::Row>(m, n, c, ldc, c_t, ldc_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##trsyl, UC##TRSYL)                                                                                                    \
        (&trana, &tranb, &isgn, &m, &n, a_t.data(), &lda_t, b_t.data(), &ldb_t, c_t.data(), &ldc_t, scale, &info);                         \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, c_t, ldc_t, c, ldc);                                                                           \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

TRSYL(double, d, D);
TRSYL(float, s, S);

#define ORGQR(Type, lc, uc)                                                                                                                \
    auto lc##orgqr(blas_int m, blas_int n, blas_int k, Type *a, blas_int lda, const Type *tau)->blas_int {                                 \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int info{0};                                                                                                                  \
        blas_int lwork{-1};                                                                                                                \
        Type     work_query;                                                                                                               \
                                                                                                                                           \
        blas_int lda_t = std::max(blas_int{1}, m);                                                                                         \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("orgqr warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##orgqr, UC##ORGQR)(&m, &n, &k, a, &lda_t, tau, &work_query, &lwork, &info);                                           \
                                                                                                                                           \
        lwork = (blas_int)work_query;                                                                                                      \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, n));                                                                            \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##orgqr, UC##ORGQR)(&m, &n, &k, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                  \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

ORGQR(double, d, D);
ORGQR(float, s, S);

#define UNGQR(Type, lc, uc)                                                                                                                \
    auto lc##ungqr(blas_int m, blas_int n, blas_int k, Type *a, blas_int lda, const Type *tau)->blas_int {                                 \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int info{0};                                                                                                                  \
        blas_int lwork{-1};                                                                                                                \
        Type     work_query;                                                                                                               \
                                                                                                                                           \
        blas_int lda_t = std::max(blas_int{1}, m);                                                                                         \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("ungqr warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##ungqr, UC##UNGQR)(&m, &n, &k, a, &lda_t, tau, &work_query, &lwork, &info);                                           \
                                                                                                                                           \
        lwork = (blas_int)(work_query.real());                                                                                             \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, n));                                                                            \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##ungqr, UC##UNGQR)(&m, &n, &k, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                  \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return info;                                                                                                                       \
    } /**/

UNGQR(std::complex<float>, c, C);
UNGQR(std::complex<double>, z, Z);

#define GEQRF(Type, lc, uc)                                                                                                                \
    auto lc##geqrf(blas_int m, blas_int n, Type *a, blas_int lda, Type *tau)->blas_int {                                                   \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int info{0};                                                                                                                  \
        blas_int lwork{-1};                                                                                                                \
        Type     work_query;                                                                                                               \
                                                                                                                                           \
        blas_int lda_t = std::max(blas_int{1}, m);                                                                                         \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geqrf warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -4;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a, &lda_t, tau, &work_query, &lwork, &info);                                               \
                                                                                                                                           \
        lwork = (blas_int)work_query;                                                                                                      \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, n));                                                                            \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                      \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

#define GEQRF_complex(Type, lc, uc)                                                                                                        \
    auto lc##geqrf(blas_int m, blas_int n, Type *a, blas_int lda, Type *tau)->blas_int {                                                   \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int info{0};                                                                                                                  \
        blas_int lwork{-1};                                                                                                                \
        Type     work_query;                                                                                                               \
                                                                                                                                           \
        blas_int lda_t = std::max(blas_int{1}, m);                                                                                         \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geqrf warning: lda < n, lda = {}, n = {}", lda, n);                                                              \
            return -4;                                                                                                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a, &lda_t, tau, &work_query, &lwork, &info);                                               \
                                                                                                                                           \
        lwork = (blas_int)(work_query.real());                                                                                             \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(blas_int{1}, n));                                                                            \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geqrf, UC##GEQRF)(&m, &n, a_t.data(), &lda_t, tau, work.data(), &lwork, &info);                                      \
                                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(m, n, a_t, lda_t, a, lda);                                                                           \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GEQRF(double, d, D);
GEQRF(float, s, S);
GEQRF_complex(std::complex<double>, z, Z);
GEQRF_complex(std::complex<float>, c, C);

#define GEEV_complex(Type, lc, UC)                                                                                                         \
    auto lc##geev(char jobvl, char jobvr, blas_int n, std::complex<Type> *a, blas_int lda, std::complex<Type> *w, std::complex<Type> *vl,  \
                  blas_int ldvl, std::complex<Type> *vr, blas_int ldvr)                                                                    \
        ->blas_int {                                                                                                                       \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int                        info  = 0;                                                                                         \
        blas_int                        lwork = -1;                                                                                        \
        std::vector<Type>               rwork;                                                                                             \
        std::vector<std::complex<Type>> work;                                                                                              \
        std::complex<Type>              work_query;                                                                                        \
                                                                                                                                           \
        /* Allocate memory for working array(s) */                                                                                         \
        rwork.resize(std::max(blas_int{1}, 2 * n));                                                                                        \
                                                                                                                                           \
        blas_int                        lda_t  = std::max(blas_int{1}, n);                                                                 \
        blas_int                        ldvl_t = std::max(blas_int{1}, n);                                                                 \
        blas_int                        ldvr_t = std::max(blas_int{1}, n);                                                                 \
        std::vector<std::complex<Type>> a_t;                                                                                               \
        std::vector<std::complex<Type>> vl_t;                                                                                              \
        std::vector<std::complex<Type>> vr_t;                                                                                              \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geev warning: lda < n, lda = {}, n = {}", lda, n);                                                               \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvl < 1 || (lsame(jobvl, 'v') && ldvl < n)) {                                                                                 \
            println_warn("geev warning: ldvl < 1 or (jobvl = 'v' and ldvl < n), ldvl = {}, n = {}", ldvl, n);                              \
            return -8;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvr < 1 || (lsame(jobvr, 'v') && ldvr < n)) {                                                                                 \
            println_warn("geev warning: ldvr < 1 or (jobvr = 'v' and ldvr < n), ldvr = {}, n = {}", ldvr, n);                              \
            return -10;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda_t, w, vl, &ldvl_t, vr, &ldvr_t, &work_query, &lwork, rwork.data(), &info);                            \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        lwork = (blas_int)work_query.real();                                                                                               \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        a_t.resize(lda_t *std::max(blas_int{1}, n));                                                                                       \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            vl_t.resize(ldvl_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            vr_t.resize(ldvr_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(n, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a_t.data(), &lda_t, w, vl_t.data(), &ldvl_t, vr_t.data(), &ldvr_t, work.data(), &lwork, rwork.data(), &info); \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(n, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vl_t, ldvl_t, vl, ldvl);                                                                   \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vr_t, ldvr_t, vr, ldvr);                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    } /**/

GEEV_complex(float, c, C);
GEEV_complex(double, z, Z);

#define GEEV(Type, lc, uc)                                                                                                                 \
    auto lc##geev(char jobvl, char jobvr, blas_int n, Type *a, blas_int lda, std::complex<Type> *w, Type *vl, blas_int ldvl, Type *vr,     \
                  blas_int ldvr)                                                                                                           \
        ->blas_int {                                                                                                                       \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        blas_int          info  = 0;                                                                                                       \
        blas_int          lwork = -1;                                                                                                      \
        std::vector<Type> work;                                                                                                            \
        Type              work_query;                                                                                                      \
                                                                                                                                           \
        blas_int lda_t  = std::max(blas_int{1}, n);                                                                                        \
        blas_int ldvl_t = std::max(blas_int{1}, n);                                                                                        \
        blas_int ldvr_t = std::max(blas_int{1}, n);                                                                                        \
                                                                                                                                           \
        std::vector<Type> a_t;                                                                                                             \
        std::vector<Type> vl_t;                                                                                                            \
        std::vector<Type> vr_t;                                                                                                            \
        std::vector<Type> wr(n), wi(n);                                                                                                    \
                                                                                                                                           \
        /* Check leading dimensions */                                                                                                     \
        if (lda < n) {                                                                                                                     \
            println_warn("geev warning: lda < n, lda = {}, n = {}", lda, n);                                                               \
            return -5;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvl < 1 || (lsame(jobvl, 'v') && ldvl < n)) {                                                                                 \
            println_warn("geev warning: ldvl < 1 or (jobvl = 'v' and ldvl < n), ldvl = {}, n = {}", ldvl, n);                              \
            return -9;                                                                                                                     \
        }                                                                                                                                  \
        if (ldvr < 1 || (lsame(jobvr, 'v') && ldvr < n)) {                                                                                 \
            println_warn("geev warning: ldvr < 1 or (jobvr = 'v' and ldvr < n), ldvr = {}, n = {}", ldvr, n);                              \
            return -11;                                                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Query optimal working array size */                                                                                             \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a, &lda_t, wr.data(), wi.data(), vl, &ldvl_t, vr, &ldvr_t, &work_query, &lwork, &info);                       \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        lwork = (blas_int)work_query;                                                                                                      \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        a_t.resize(lda_t *std::max(blas_int{1}, n));                                                                                       \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            vl_t.resize(ldvl_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            vr_t.resize(ldvr_t *std::max(blas_int{1}, n));                                                                                 \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(n, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call LAPACK function and adjust info */                                                                                         \
        FC_GLOBAL(lc##geev, UC##GEEV)                                                                                                      \
        (&jobvl, &jobvr, &n, a_t.data(), &lda_t, wr.data(), wi.data(), vl_t.data(), &ldvl_t, vr_t.data(), &ldvr_t, work.data(), &lwork,    \
         &info);                                                                                                                           \
        if (info < 0) {                                                                                                                    \
            return info;                                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose output matrices */                                                                                                    \
        transpose<OrderMajor::Column>(n, n, a_t, lda_t, a, lda);                                                                           \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vl_t, ldvl_t, vl, ldvl);                                                                   \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            transpose<OrderMajor::Column>(n, n, vr_t, ldvr_t, vr, ldvr);                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Pack wr and wi into w */                                                                                                        \
        for (blas_int i = 0; i < n; i++) {                                                                                                 \
            w[i] = std::complex<float>(wr[i], wi[i]);                                                                                      \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    }                                                                                                                                      \
    /**/

GEEV(float, s, S);
GEEV(double, d, D);

END_EINSUMS_NAMESPACE_CPP(einsums::backend::vendor)