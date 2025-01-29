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
extern void FC_GLOBAL(dgees, DGEES)(char *, char *, int_t (*)(double *, double *), int_t *, double *, int_t *, int_t *, double *, double *,
                                    double *, int_t *, double *, int_t *, int_t *, int_t *);
extern void FC_GLOBAL(sgees, SGEES)(char *, char *, int_t (*)(float *, float *), int_t *, float *, int_t *, int_t *, float *, float *,
                                    float *, int_t *, float *, int_t *, int_t *, int_t *);
}

#define GEES(Type, lc, UC)                                                                                                                 \
    auto lc##gees(char jobvs, int_t n, Type *a, int_t lda, int_t *sdim, Type *wr, Type *wi, Type *vs, int_t ldvs) -> int_t {               \
        LabeledSection0();                                                                                                                 \
                                                                                                                                           \
        int_t  info  = 0;                                                                                                                  \
        int_t  lwork = -1;                                                                                                                 \
        int_t *bwork = nullptr;                                                                                                            \
                                                                                                                                           \
        Type work_query;                                                                                                                   \
                                                                                                                                           \
        int_t lda_t  = std::max(int_t{1}, n);                                                                                              \
        int_t ldvs_t = std::max(int_t{1}, n);                                                                                              \
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
        lwork = (int_t)work_query;                                                                                                         \
        /* Allocate memory for work array */                                                                                               \
        std::vector<Type> work(lwork);                                                                                                     \
                                                                                                                                           \
        /* Allocate memory for temporary array(s) */                                                                                       \
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, n));                                                                               \
        std::vector<Type> vs_t;                                                                                                            \
        if (lsame(jobvs, 'v')) {                                                                                                           \
            vs_t.resize(ldvs_t *std::max(int_t{1}, n));                                                                                    \
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

} // namespace einsums::blas::vendor