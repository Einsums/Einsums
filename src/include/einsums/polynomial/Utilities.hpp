//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/STL.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"

#include <algorithm>

namespace einsums::polynomial {

template <template <typename, size_t> typename TensorType, typename T>
auto get_domain(const TensorType<T, 1> &x) -> Tensor<T, 1> {
    T min{0}, max{0};
    T imin{0}, imax{0};

    if constexpr (IsComplexV<T>) {
        min = max = x(0).real();
        imin = imax = x(0).imag();
    } else {
        min = max = x(0);
    }

    EINSUMS_OMP_PARALLEL {
        T    local_min{0}, local_max{0};
        T    local_imin{0}, local_imax{0};
        auto size = x.size();

        if constexpr (IsComplexV<T>) {
            local_min = local_max = x(0).real();
            local_imin = local_imax = x(0).imag();
        } else {
            local_min = local_max = x(0);
        }

        EINSUMS_OMP_FOR_NOWAIT
        for (size_t i = 1; i < size; i++) {
            if constexpr (IsComplexV<T>) {
                if (x(i).real() < local_min)
                    local_min = x(i).real();
                if (x(i).imag() < local_imin)
                    local_imin = x(i).imag();
                if (x(i).real() > local_max)
                    local_max = x(i).real();
                if (x(i).imag() > local_imax)
                    local_imax = x(i).imag();
            } else {
                if (x(i) < local_min)
                    local_min = x(i);
                if (x(i) > local_max)
                    local_max = x(i);
            }
        }

        EINSUMS_OMP_CRITICAL {
            if (local_min < min)
                min = local_min;
            if (local_imin < imin)
                imin = local_imin;
            if (local_max > max)
                max = local_max;
            if (local_imax > imax)
                imax = local_imax;
        }
    }

    auto result = create_tensor<T>("domain", 2);
    if constexpr (IsComplexV<T>) {
        result(0) = T{min, imin};
        result(1) = T{max, imax};
    } else {
        result(0) = min;
        result(1) = max;
    }
    return result;
}

namespace detail {
template <typename T, typename LineFunction, typename MultiplyFunction>
auto from_roots(LineFunction line, MultiplyFunction mult, const Tensor<T, 1> &roots) -> Tensor<T, 1> {
    if (roots.dim(0) == 0) {
        return create_ones_tensor("from_roots", 1);
    } else {
        auto r = create_tensor_like(roots);
        auto p = create_tensor_like(roots);

        r       = roots;
        auto rd = r.vector_data();
        std::sort(rd.begin(), rd.end(), std::greater<T>());

        auto pd = p.vector_data();
        auto n  = pd.size();
        for (auto [i, val] : enumerate(rd)) {
            pd[i] = line(-val, 1);
        }

        while (n > 1) {
            auto res = std::div(n, 2);
            auto a = res.rem;
            auto b = res.quot;
        }
    }
}
} // namespace detail
} // namespace einsums::polynomial