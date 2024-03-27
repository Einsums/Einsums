//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cctype>
#include <vector>

namespace einsums::backend::linear_algebra::vendor::util {

inline bool lsame(char ca, char cb) {
    return std::tolower(ca) == std::tolower(cb);
}

enum OrderMajor { Column, Row };

// OrderMajor indicates the order of the input matrix. C is Row, Fortran is Column
template <OrderMajor Order, typename T, typename Integer>
void transpose(Integer m, Integer n, const T *in, Integer ldin, T *out, Integer ldout) {
    Integer i, j, x, y;

    if (in == nullptr || out == nullptr) {
        return;
    }

    if constexpr (Order == OrderMajor::Column) {
        x = n;
        y = m;
    } else if constexpr (Order == OrderMajor::Row) {
        x = m;
        y = n;
    } else {
        static_assert(Order == OrderMajor::Column || Order == OrderMajor::Row, "Invalid OrderMajor");
    }

    // Look into replacing this with hptt or librett
    for (i = 0; i < std::min(y, ldin); i++) {
        for (j = 0; j < std::min(x, ldout); j++) {
            out[(size_t)i * ldout + j] = in[(size_t)j * ldin + i];
        }
    }
}

template <OrderMajor Order, typename T, typename Integer>
void transpose(Integer m, Integer n, const std::vector<T> &in, Integer ldin, T *out, Integer ldout) {
    tranpose<Order>(m, n, in.data(), ldin, out, ldout);
}

template <OrderMajor Order, typename T, typename Integer>
void transpose(Integer m, Integer n, const T *in, Integer ldin, std::vector<T> &out, Integer ldout) {
    tranpose<Order>(m, n, in, ldin, out.data(), ldout);
}

} // namespace einsums::backend::linear_algebra::vendor::util