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
extern void FC_GLOBAL(sgeev, SGEEV)(char *, char *, int_t *, float *, int_t *, float *, float *, float *, int_t *, float *, int_t *,
                                    float *, int_t *, int_t *);
extern void FC_GLOBAL(dgeev, DGEEV)(char *, char *, int_t *, double *, int_t *, double *, double *, double *, int_t *, double *, int_t *,
                                    double *, int_t *, int_t *);
extern void FC_GLOBAL(cgeev, CGEEV)(char *, char *, int_t *, std::complex<float> *, int_t *, std::complex<float> *, std::complex<float> *,
                                    int_t *, std::complex<float> *, int_t *, std::complex<float> *, int_t *, float *, int_t *);
extern void FC_GLOBAL(zgeev, ZGEEV)(char *, char *, int_t *, std::complex<double> *, int_t *, std::complex<double> *,
                                    std::complex<double> *, int_t *, std::complex<double> *, int_t *, std::complex<double> *, int_t *,
                                    double *, int_t *);
}

#define GEEV_complex(Type, lc, UC)                                                                                                         \
    auto lc##geev(char jobvl, char jobvr, int_t n, std::complex<Type> *a, int_t lda, std::complex<Type> *w, std::complex<Type> *vl,        \
                  int_t ldvl, std::complex<Type> *vr, int_t ldvr)                                                                          \
        ->int_t {                                                                                                                          \
        EINSUMS_PROFILE_SCOPE("BLASVendor");                                                                                               \
                                                                                                                                           \
        int_t                           info  = 0;                                                                                         \
        int_t                           lwork = -1;                                                                                        \
        std::vector<Type>               rwork;                                                                                             \
        std::vector<std::complex<Type>> work;                                                                                              \
        std::complex<Type>              work_query;                                                                                        \
                                                                                                                                           \
        /* Allocate memory for working array(s) */                                                                                         \
        rwork.resize(std::max(int_t{1}, 2 * n));                                                                                           \
                                                                                                                                           \
        int_t                           lda_t  = std::max(int_t{1}, n);                                                                    \
        int_t                           ldvl_t = std::max(int_t{1}, n);                                                                    \
        int_t                           ldvr_t = std::max(int_t{1}, n);                                                                    \
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
        lwork = (int_t)work_query.real();                                                                                                  \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        a_t.resize(lda_t * std::max(int_t{1}, n));                                                                                         \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            vl_t.resize(ldvl_t * std::max(int_t{1}, n));                                                                                   \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            vr_t.resize(ldvr_t * std::max(int_t{1}, n));                                                                                   \
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
    auto lc##geev(char jobvl, char jobvr, int_t n, Type *a, int_t lda, std::complex<Type> *w, Type *vl, int_t ldvl, Type *vr, int_t ldvr)  \
        ->int_t {                                                                                                                          \
        EINSUMS_PROFILE_SCOPE("BLASVendor");                                                                                               \
                                                                                                                                           \
        int_t             info  = 0;                                                                                                       \
        int_t             lwork = -1;                                                                                                      \
        std::vector<Type> work;                                                                                                            \
        Type              work_query;                                                                                                      \
                                                                                                                                           \
        int_t lda_t  = std::max(int_t{1}, n);                                                                                              \
        int_t ldvl_t = std::max(int_t{1}, n);                                                                                              \
        int_t ldvr_t = std::max(int_t{1}, n);                                                                                              \
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
        lwork = (int_t)work_query;                                                                                                         \
        work.resize(lwork);                                                                                                                \
                                                                                                                                           \
        a_t.resize(lda_t * std::max(int_t{1}, n));                                                                                         \
        if (lsame(jobvl, 'v')) {                                                                                                           \
            vl_t.resize(ldvl_t * std::max(int_t{1}, n));                                                                                   \
        }                                                                                                                                  \
        if (lsame(jobvr, 'v')) {                                                                                                           \
            vr_t.resize(ldvr_t * std::max(int_t{1}, n));                                                                                   \
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
        for (int_t i = 0; i < n; i++) {                                                                                                    \
            w[i] = std::complex<float>(wr[i], wi[i]);                                                                                      \
        }                                                                                                                                  \
                                                                                                                                           \
        return 0;                                                                                                                          \
    }                                                                                                                                      \
    /**/

GEEV(float, s, S);
GEEV(double, d, D);

} // namespace einsums::blas::vendor