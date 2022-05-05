#include "mkl.hpp"

#include "einsums/Print.hpp"
#include "fmt/format.h"

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

namespace einsums::backend::mkl {

namespace {

std::vector<sycl::device *> g_Devices;

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
    println("Unknown transpose code {}, default to CblasNoTrans.", transpose);
    return CblasNoTrans;
}

} // namespace

void initialize() {
    auto platforms = sycl::platform::get_platforms();

    for (auto platform : platforms) {
    }
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, // NOLINT
           double beta, double *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_dgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, // NOLINT
           int incy) {
    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::vendor::dgemv transa argument is invalid.");

    cblas_dgemv(CblasRowMajor, TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
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
        throw std::runtime_error(fmt::format("einsums::backend::vendor::dger: m ({}) is less than zero.", m));
    } else if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::dger: n ({}) is less than zero.", n));
    } else if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::dger: inc_x ({}) is zero.", inc_x));
    } else if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::dger: inc_y ({}) is zero.", inc_y));
    } else if (lda < std::max(1, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::vendor::dger: lda ({}) is less than max(1, n ({})).", lda, n));
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

} // namespace einsums::backend::mkl
