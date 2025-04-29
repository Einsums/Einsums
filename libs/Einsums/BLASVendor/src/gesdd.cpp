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
extern void FC_GLOBAL(dgesdd, DGESDD)(char *, int_t *, int_t *, double *, int_t *, double *, double *, int_t *, double *, int_t *, double *,
                                      int_t *, int_t *, int_t *);
extern void FC_GLOBAL(sgesdd, SGESDD)(char *, int_t *, int_t *, float *, int_t *, float *, float *, int_t *, float *, int_t *, float *,
                                      int_t *, int_t *, int_t *);
extern void FC_GLOBAL(zgesdd, ZGESDD)(char *, int_t *, int_t *, std::complex<double> *, int_t *, double *, std::complex<double> *, int_t *,
                                      std::complex<double> *, int_t *, std::complex<double> *, int_t *, double *, int_t *, int_t *);
extern void FC_GLOBAL(cgesdd, CGESDD)(char *, int_t *, int_t *, std::complex<float> *, int_t *, float *, std::complex<float> *, int_t *,
                                      std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *, int_t *);
}

#define GESDD(Type, lcletter, UCLETTER)                                                                                                    \
    auto lcletter##gesdd(char jobz, int_t m, int_t n, Type *a, int_t lda, Type *s, Type *u, int_t ldu, Type *vt, int_t ldvt)->int_t {      \
        EINSUMS_PROFILE_SCOPE("BLASVendor");                                                                                               \
        int_t nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;                                    \
        int_t ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);                  \
        int_t nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);                 \
                                                                                                                                           \
        int_t             lda_t  = std::max(int_t{1}, m);                                                                                  \
        int_t             ldu_t  = std::max(int_t{1}, nrows_u);                                                                            \
        int_t             ldvt_t = std::max(int_t{1}, nrows_vt);                                                                           \
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
        int_t info{0};                                                                                                                     \
        int_t lwork{-1};                                                                                                                   \
        Type  work_query;                                                                                                                  \
        FC_GLOBAL(lcletter##gesdd, UCLETTER##GESDD)                                                                                        \
        (&jobz, &m, &n, a, &lda_t, s, u, &ldu_t, vt, &ldvt_t, &work_query, &lwork, nullptr, &info);                                        \
        lwork = (int)work_query;                                                                                                           \
                                                                                                                                           \
        /* Allocate memory for temporary arrays(s) */                                                                                      \
        a_t.resize(lda_t * std::max(int_t{1}, n));                                                                                         \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            u_t.resize(ldu_t * std::max(int_t{1}, ncols_u));                                                                               \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            vt_t.resize(ldvt_t * std::max(int_t{1}, n));                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        /* Allocate work array */                                                                                                          \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate iwork array */                                                                                                         \
        std::vector<int_t> iwork(8 * std::min(m, n));                                                                                      \
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
    auto lc##gesdd(char jobz, int_t m, int_t n, std::complex<Type> *a, int_t lda, Type *s, std::complex<Type> *u, int_t ldu,               \
                   std::complex<Type> *vt, int_t ldvt)                                                                                     \
        ->int_t {                                                                                                                          \
        EINSUMS_PROFILE_SCOPE("BLASVendor");                                                                                               \
        int_t nrows_u  = (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, '0') && m < n)) ? m : 1;                                    \
        int_t ncols_u  = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m < n)) ? m : (lsame(jobz, 's') ? std::min(m, n) : 1);                  \
        int_t nrows_vt = (lsame(jobz, 'a') || (lsame(jobz, 'o') && m >= n)) ? n : (lsame(jobz, 's') ? std::min(m, n) : 1);                 \
                                                                                                                                           \
        int_t                           lda_t  = std::max(int_t{1}, m);                                                                    \
        int_t                           ldu_t  = std::max(int_t{1}, nrows_u);                                                              \
        int_t                           ldvt_t = std::max(int_t{1}, nrows_vt);                                                             \
        int_t                           info{0};                                                                                           \
        int_t                           lwork{-1};                                                                                         \
        size_t                          lrwork;                                                                                            \
        std::complex<Type>              work_query;                                                                                        \
        std::vector<std::complex<Type>> a_t, u_t, vt_t;                                                                                    \
        std::vector<Type>               rwork;                                                                                             \
        std::vector<std::complex<Type>> work;                                                                                              \
        std::vector<int_t>              iwork;                                                                                             \
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
            lrwork = std::max(int_t{1}, 7 * std::min(m, n));                                                                               \
        } else {                                                                                                                           \
            lrwork = (size_t)std::max(int_t{1},                                                                                            \
                                      std::min(m, n) * std::max(5 * std::min(m, n) + 7, 2 * std::max(m, n) + 2 * std::min(m, n) + 1));     \
        }                                                                                                                                  \
                                                                                                                                           \
        iwork.resize(std::max(int_t{1}, 8 * std::min(m, n)));                                                                              \
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
        a_t.resize(lda_t * std::max(int_t{1}, n));                                                                                         \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m < n))) {                                                       \
            u_t.resize(ldu_t * std::max(int_t{1}, ncols_u));                                                                               \
        }                                                                                                                                  \
        if (lsame(jobz, 'a') || lsame(jobz, 's') || (lsame(jobz, 'o') && (m >= n))) {                                                      \
            vt_t.resize(ldvt_t * std::max(int_t{1}, n));                                                                                   \
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

} // namespace einsums::blas::vendor