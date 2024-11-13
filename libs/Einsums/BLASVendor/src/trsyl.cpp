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
extern void FC_GLOBAL(dtrsyl, DTRSYL)(char *, char *, int_t *, int_t *, int_t *, double const *, int_t *, double const *, int_t *, double *,
                                      int_t *, double *, int_t *);
extern void FC_GLOBAL(strsyl, STRSYL)(char *, char *, int_t *, int_t *, int_t *, float const *, int_t *, float const *, int_t *, float *,
                                      int_t *, float *, int_t *);
}

#define TRSYL(Type, lc, uc)                                                                                                                \
    auto lc##trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const Type *a, int_t lda, const Type *b, int_t ldb, Type *c,      \
                   int_t ldc, Type *scale) -> int_t {                                                                                      \
        int_t info  = 0;                                                                                                                   \
        int_t lda_t = std::max(int_t{1}, m);                                                                                               \
        int_t ldb_t = std::max(int_t{1}, n);                                                                                               \
        int_t ldc_t = std::max(int_t{1}, m);                                                                                               \
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
        std::vector<Type> a_t(lda_t *std::max(int_t{1}, m));                                                                               \
        std::vector<Type> b_t(ldb_t *std::max(int_t{1}, n));                                                                               \
        std::vector<Type> c_t(ldc_t *std::max(int_t{1}, n));                                                                               \
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

} // namespace einsums::blas::vendor