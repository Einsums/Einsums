#include "EinsumsInCpp/Blas.hpp"

#include <stdexcept>

#ifndef FC_SYMBOL
#define FC_SYMBOL 2
#endif

#if FC_SYMBOL == 1
/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name, NAME) name
#elif FC_SYMBOL == 2
/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL(name, NAME) name##_
#elif FC_SYMBOL == 3
/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name, NAME) NAME
#elif FC_SYMBOL == 4
/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL(name, NAME) NAME##_
#endif

extern "C" {
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int *, int *, int *, double *, const double *, int *, const double *, int *, double *,
                                    double *, int *);
extern void FC_GLOBAL(dgemv, DGEMV)(char *, int *, int *, double *, const double *, int *, const double *, int *, double *, double *,
                                    int *);
extern void FC_GLOBAL(dsyev, DSYEV)(char *, char *, int *, double *, int *, double *, double *, int *, int *);
extern void FC_GLOBAL(dgesv, DGESV)(int *, int *, double *, int *, int *, double *, int *, int *);
extern void FC_GLOBAL(dscal, DSCAL)(int *, double *, double *, int *);
extern double FC_GLOBAL(ddot, DDOT)(int *, const double *, int *, const double *, int *); // NOLINT
extern void FC_GLOBAL(daxpy, DAXPY)(int *, double *, const double *, int *, double *, int *);
extern void FC_GLOBAL(dger, DGER)(int *, int *, double *, const double *, int *, const double *, int *, double *, int *);
}

namespace EinsumsInCpp::Blas {

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc) {
    if (m == 0 || n == 0 || k == 0)
        return;
    FC_GLOBAL(dgemm, DGEMM)(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    if (m == 0 || n == 0)
        return;
    if (transa == 'N' || transa == 'n')
        transa = 'T';
    else if (transa == 'T' || transa == 't')
        transa = 'N';
    else
        throw std::invalid_argument("EinsumsInCpp::Blas::dgemv transa argument is invalid.");

    FC_GLOBAL(dgemv, DGEMV)(&transa, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int {
    int info{0};
    FC_GLOBAL(dsyev, DSYEV)(&job, &uplo, &n, a, &lda, w, work, &lwork, &info);
    return info;
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    int info{0};
    FC_GLOBAL(dgesv, DGESV)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

void dscal(int n, double alpha, double *vec, int inc) { FC_GLOBAL(dscal, DSCAL)(&n, &alpha, vec, &inc); }

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double { return FC_GLOBAL(ddot, DDOT)(&n, x, &incx, y, &incy); }

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    FC_GLOBAL(daxpy, DAXPY)(&n, &alpha_x, x, &inc_x, y, &inc_y);
}

void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    FC_GLOBAL(dger, DGER)(&n, &m, &alpha, y, &inc_y, x, &inc_x, a, &lda);
}

} // namespace EinsumsInCpp::Blas