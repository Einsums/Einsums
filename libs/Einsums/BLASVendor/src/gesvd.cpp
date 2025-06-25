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
extern void FC_GLOBAL(dgesvd, DGESVD)(char *, char *, int_t *, int_t *, double *, int_t *, double *, double *, int_t *, double *, int_t *,
                                      double *, int_t *, int_t *);
extern void FC_GLOBAL(sgesvd, SGESVD)(char *, char *, int_t *, int_t *, float *, int_t *, float *, float *, int_t *, float *, int_t *,
                                      float *, int_t *, int_t *);
extern void FC_GLOBAL(zgesvd, ZGESVD)(char *jobu, char *jobvt, int_t *m, int_t *n, std::complex<double> *A, int_t *lda, double *S,
                                      std::complex<double> *U, int_t *ldu, std::complex<double> *Vt, int_t *ldvt,
                                      std::complex<double> *work, int_t *lwork, double *rwork, int_t *info);
extern void FC_GLOBAL(cgesvd, CGESVD)(char *jobu, char *jobvt, int_t *m, int_t *n, std::complex<float> *A, int_t *lda, float *S,
                                      std::complex<float> *U, int_t *ldu, std::complex<float> *Vt, int_t *ldvt, std::complex<float> *work,
                                      int_t *lwork, float *rwork, int_t *info);
}

#define GESVD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesvd(char jobu, char jobvt, int_t m, int_t n, Type *a, int_t lda, Type *s, Type *u, int_t ldu, Type *vt, int_t ldvt,   \
                         Type *superb)                                                                                                     \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info  = 0;                                                                                                                   \
        int_t lwork = -1;                                                                                                                  \
                                                                                                                                           \
        Type  work_query;                                                                                                                  \
        int_t i;                                                                                                                           \
                                                                                                                                           \
        int_t nrows_u  = (lsame(jobu, 'a') || lsame(jobu, 's')) ? m : 1;                                                                   \
        int_t ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std::min(m, n) : 1);                                                   \
        int_t nrows_vt = lsame(jobvt, 'a') ? n : (lsame(jobvt, 's') ? std::min(m, n) : 1);                                                 \
        int_t ncols_vt = (lsame(jobvt, 'a') || lsame(jobvt, 's')) ? n : 1;                                                                 \
                                                                                                                                           \
        int_t lda_t  = std::max(int_t{1}, m);                                                                                              \
        int_t ldu_t  = std::max(int_t{1}, nrows_u);                                                                                        \
        int_t ldvt_t = std::max(int_t{1}, nrows_vt);                                                                                       \
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
        lwork = (int_t)work_query;                                                                                                         \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
        std::vector<Type> u_t, vt_t;                                                                                                       \
        if (lsame(jobu, 'a') || lsame(jobu, 's')) {                                                                                        \
            u_t.resize(ldu_t *std::max(int_t{1}, ncols_u));                                                                                \
        }                                                                                                                                  \
        if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {                                                                                      \
            vt_t.resize(ldvt_t *std::max(int_t{1}, n));                                                                                    \
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

auto dgesvd(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt,
            double *superb) -> int_t {
    Section const _section(fmt ::format("{}", __func__));
    int_t         info  = 0;
    int_t         lwork = -1;
    double        work_query;
    int_t         i;
    int_t         nrows_u  = (lsame(jobu, 'a') || lsame(jobu, 's')) ? m : 1;
    int_t         ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std ::min(m, n) : 1);
    int_t         nrows_vt = lsame(jobvt, 'a') ? n : (lsame(jobvt, 's') ? std ::min(m, n) : 1);
    int_t         ncols_vt = (lsame(jobvt, 'a') || lsame(jobvt, 's')) ? n : 1;
    int_t         lda_t    = std ::max(int_t{1}, m);
    int_t         ldu_t    = std ::max(int_t{1}, nrows_u);
    int_t         ldvt_t   = std ::max(int_t{1}, nrows_vt);
    if (lda < n) {
        println_warn("gesvd warning: lda < n, lda = {}, n = {}", lda, n);
        return -6;
    }
    if (ldu < ncols_u) {
        println_warn("gesvd warning: ldu < ncols_u, ldu = {}, ncols_u = {}", ldu, ncols_u);
        return -9;
    }
    if (ldvt < ncols_vt) {
        println_warn("gesvd warning: ldvt < ncols_vt, ldvt = {}, ncols_vt = {}", ldvt, ncols_vt);
        return -11;
    }
    dgesvd_(&jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, &info);
    if (info != 0)
        println_abort("gesvd work array size query failed. info {}", info);
    lwork = (int_t)work_query;
    std ::vector<double> work(lwork);
    std ::vector<double> a_t(lda_t * std ::max(int_t{1}, n));
    std ::vector<double> u_t, vt_t;
    if (lsame(jobu, 'a') || lsame(jobu, 's')) {
        u_t.resize(ldu_t * std ::max(int_t{1}, ncols_u));
    }
    if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {
        vt_t.resize(ldvt_t * std ::max(int_t{1}, n));
    }
    transpose<OrderMajor ::Row>(m, n, a, lda, a_t, lda_t);
    dgesvd_(&jobu, &jobvt, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, &info);
    if (info < 0) {
        println_abort("gesvd lapack routine failed. info {}", info);
        return info;
    }
    transpose<OrderMajor ::Column>(m, n, a_t, lda_t, a, lda);
    if (lsame(jobu, 'a') || lsame(jobu, 's')) {
        transpose<OrderMajor ::Column>(nrows_u, ncols_u, u_t, ldu_t, u, ldu);
    }
    if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {
        transpose<OrderMajor ::Column>(nrows_vt, n, vt_t, ldvt_t, vt, ldvt);
    }
    for (i = 0; i < std::min(std ::min(m, n), lwork) - 1; i++) {
        superb[i] = work[i + 1];
    }
    return 0;
};
GESVD(float, s, S);

#define GESVD_complex(Type, lcletter, UCLETTER)                                                                                            \
    auto lcletter##gesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<Type> *a, int_t lda, Type *s, std::complex<Type> *u,        \
                         int_t ldu, std::complex<Type> *vt, int_t ldvt, std::complex<Type> *superb)                                        \
        ->int_t {                                                                                                                          \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t info  = 0;                                                                                                                   \
        int_t lwork = -1;                                                                                                                  \
                                                                                                                                           \
        std::complex<Type> work_query;                                                                                                     \
        int_t              i;                                                                                                              \
                                                                                                                                           \
        int_t nrows_u  = (lsame(jobu, 'a') || lsame(jobu, 's')) ? m : 1;                                                                   \
        int_t ncols_u  = lsame(jobu, 'a') ? m : (lsame(jobu, 's') ? std::min(m, n) : 1);                                                   \
        int_t nrows_vt = lsame(jobvt, 'a') ? n : (lsame(jobvt, 's') ? std::min(m, n) : 1);                                                 \
        int_t ncols_vt = (lsame(jobvt, 'a') || lsame(jobvt, 's')) ? n : 1;                                                                 \
                                                                                                                                           \
        int_t lda_t  = std::max(int_t{1}, m);                                                                                              \
        int_t ldu_t  = std::max(int_t{1}, nrows_u);                                                                                        \
        int_t ldvt_t = std::max(int_t{1}, nrows_vt);                                                                                       \
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
        std::vector<Type> rwork(5 * std::min(m, n));                                                                                       \
        /* Query optimal working array(s) size */                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, rwork.data(), &info);                           \
        if (info != 0)                                                                                                                     \
            println_abort("gesvd work array size query failed. info {}", info);                                                            \
                                                                                                                                           \
        lwork = (int_t)work_query.real();                                                                                                  \
                                                                                                                                           \
        /* Allocate memory for work array */                                                                                               \
        std::vector<std::complex<Type>> work(lwork);                                                                                       \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<std::complex<Type>> a_t(lda_t *std::max(int_t{1}, n));                                                                 \
        std::vector<std::complex<Type>> u_t, vt_t;                                                                                         \
        if (lsame(jobu, 'a') || lsame(jobu, 's')) {                                                                                        \
            u_t.resize(ldu_t *std::max(int_t{1}, ncols_u));                                                                                \
        }                                                                                                                                  \
        if (lsame(jobvt, 'a') || lsame(jobvt, 's')) {                                                                                      \
            vt_t.resize(ldvt_t *std::max(int_t{1}, n));                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Transpose input matrices */                                                                                                     \
        transpose<OrderMajor::Row>(m, n, a, lda, a_t, lda_t);                                                                              \
                                                                                                                                           \
        /* Call lapack routine */                                                                                                          \
        FC_GLOBAL(lcletter##gesvd, UCLETTER##GESVD)                                                                                        \
        (&jobu, &jobvt, &m, &n, a_t.data(), &lda_t, s, u_t.data(), &ldu_t, vt_t.data(), &ldvt_t, work.data(), &lwork, rwork.data(),        \
         &info);                                                                                                                           \
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

GESVD_complex(double, z, Z);
GESVD_complex(float, c, C);

} // namespace einsums::blas::vendor