#include "einsums/Print.hpp"
#include "fmt/format.h"

#include <exception>

#if defined(EINSUMS_HAVE_LAPACKE)
#include <cblas.h>
#include <lapacke.h>
#endif

#if defined(EINSUMS_HAVE_MKL_LAPACKE)
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#endif

namespace einsums::backend::cblas {

namespace {

auto transpose_to_cblas(char transpose) -> CBLAS_TRANSPOSE {
    switch (transpose) {
    case 'N':
    case 'n':
        return CblasNoTrans;
    case 'T':
    case 't':
        return CblasTrans;
    case 'C':
    case 'c':
        return CblasConjTrans;
    }
    println_warn("Unknown transpose code {}, defaulting to CblasNoTrans.", transpose);
    return CblasNoTrans;
}

} // namespace

void initialize() {
}

void finalize() {
}

void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, // NOLINT
           double beta, double *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_dgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_cgemm(CblasRowMajor, TransA, TransB, m, n, k, static_cast<void *>(&alpha), static_cast<const void *>(a), lda,
                static_cast<const void *>(b), ldb, static_cast<void *>(&beta), static_cast<void *>(c), ldc);
}

void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_zgemm(CblasRowMajor, TransA, TransB, m, n, k, static_cast<void *>(&alpha), static_cast<const void *>(a), lda,
                static_cast<const void *>(b), ldb, static_cast<void *>(&beta), static_cast<void *>(c), ldc);
}

void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_sgemv(CblasRowMajor, TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_dgemv(CblasRowMajor, TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_cgemv(CblasRowMajor, TransA, m, n, static_cast<const void *>(&alpha), static_cast<const void *>(a), lda, x, incx,
                static_cast<const void *>(&beta), static_cast<void *>(y), incy);
}

void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy) {
    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_zgemv(CblasRowMajor, TransA, m, n, static_cast<const void *>(&alpha), static_cast<const void *>(a), lda, x, incx,
                static_cast<const void *>(&beta), static_cast<void *>(y), incy);
}

auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *, int) -> int {
    return LAPACKE_ssyev(LAPACK_ROW_MAJOR, job, uplo, n, a, lda, w);
}

auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *, int) -> int {
    return LAPACKE_dsyev(LAPACK_ROW_MAJOR, job, uplo, n, a, lda, w);
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
}

void dscal(int n, double alpha, double *vec, int inc) {
    cblas_dscal(n, alpha, vec, inc);
}

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double {
    return cblas_ddot(n, x, incx, y, incy);
}

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    cblas_daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: m ({}) is less than zero.", m));
    } else if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: n ({}) is less than zero.", n));
    } else if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: inc_x ({}) is zero.", inc_x));
    } else if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: inc_y ({}) is zero.", inc_y));
    } else if (lda < std::max(1, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: lda ({}) is less than max(1, n ({})).", lda, n));
    }

    cblas_dger(CblasRowMajor, m, n, alpha, y, inc_y, x, inc_x, a, lda);
}

auto dgetrf(int m, int n, double *a, int lda, int *ipiv) -> int {
    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
}

auto dgetri(int n, double *a, int lda, const int *ipiv, double *, int) -> int {
    return LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, lda, (int *)ipiv);
}

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *) -> double {
    return LAPACKE_dlange(LAPACK_ROW_MAJOR, norm_type, m, n, A, lda);
}

auto dgesdd(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *, int, int *) -> int {
    return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgees(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int {
    return LAPACKE_dgees(LAPACK_ROW_MAJOR, jobvs, 'N', nullptr, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto dtrsyl(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c, int ldc,
            double *scale) -> int {
    return LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

} // namespace einsums::backend::cblas
