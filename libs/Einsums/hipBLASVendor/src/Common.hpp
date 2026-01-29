//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Assert.hpp>
#include <Einsums/BLAS/Types.hpp>
#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/Errors.hpp>
#include <hipblas/hipblas.h>

#include <omp.h>

namespace einsums::blas::vendor {

inline bool lsame(char ca, char cb) {
    return std::tolower(ca) == std::tolower(cb);
}

enum class OrderMajor { Column, Row, C = Row, Fortran = Column };

// OrderMajor indicates the order of the input matrix. C is Row, Fortran is Column
template <OrderMajor Order, typename T>
void transpose(int m, int n, T const *in, int ldin, T *out, int ldout) {
    if (in == nullptr || out == nullptr) {
        return;
    }
    EINSUMS_ASSERT(m >= 0);
    EINSUMS_ASSERT(n >= 0);

    int    perm[]    = {1, 0};
    size_t size_in[] = {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, outer_size_in[2], outer_size_out[2];
    if constexpr (Order == OrderMajor::Row) {
        outer_size_in[0]  = static_cast<unsigned long>(m);
        outer_size_in[1]  = static_cast<unsigned long>(ldin);
        outer_size_out[0] = static_cast<unsigned long>(n);
        outer_size_out[1] = static_cast<unsigned long>(ldout);
    } else {
        outer_size_in[0]  = static_cast<unsigned long>(ldin);
        outer_size_in[1]  = static_cast<unsigned long>(n);
        outer_size_out[0] = static_cast<unsigned long>(ldout);
        outer_size_out[1] = static_cast<unsigned long>(m);
    }

    auto plan = hptt::create_plan(perm, 2, T{1.0}, in, size_in, outer_size_in, T{0.0}, out, outer_size_out, hptt::ESTIMATE,
                                  omp_get_max_threads(), {}, Order == OrderMajor::Row);

    plan->execute();
}

template <OrderMajor Order, typename T, typename Alloc1>
void transpose(int m, int n, std::vector<T, Alloc1> const &in, int ldin, T *out, int ldout) {
    transpose<Order>(m, n, in.data(), ldin, out, ldout);
}

template <OrderMajor Order, typename T, typename Alloc2>
void transpose(int m, int n, T const *in, int ldin, std::vector<T, Alloc2> &out, int ldout) {
    transpose<Order>(m, n, in, ldin, out.data(), ldout);
}

template <OrderMajor Order, typename T, typename Alloc1, typename Alloc2>
void transpose(int m, int n, std::vector<T, Alloc1> const &in, int ldin, std::vector<T, Alloc2> &out, int ldout) {
    transpose<Order>(m, n, in.data(), ldin, out.data(), ldout);
}

inline hipblasOperation_t char_to_op(char trans) {
    switch (trans) {
    case 'c':
    case 'C':
        return HIPBLAS_OP_C;
    case 'n':
    case 'N':
        return HIPBLAS_OP_N;
    case 't':
    case 'T':
        return HIPBLAS_OP_T;
    default:
        EINSUMS_THROW_EXCEPTION(enum_error, "Transpose value invalid! Expected c, n, or t case insensitive, got {}.", trans);
    }
}
} // namespace einsums::blas::vendor