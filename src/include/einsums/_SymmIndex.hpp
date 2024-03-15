#pragma once

#include "einsums/_Common.hpp"

#include <algorithm>
#include <numeric>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums)

template <typename... Idx>
struct SymmIndex {};

template <typename... Idx>
struct AntiSymmIndex {};

template <int k>
inline size_t choose_up(size_t n) {
    if constexpr (k == 1) {
        return n;
    } else if constexpr (k == 2) {
        return (n * (n + 1)) / 2;
    } else {

        size_t num = 1, den = 1;

#pragma unroll
        for (int i = 0; i < k; i++) {
            num *= n + i;
            den *= i + 1;

            auto common = std::gcd(num, den);

            num /= common;
            den /= common;
        }

        // den will be 1 at the end of this.
        return num;
    }
}

template <typename... MultiIndex>
size_t symm_index_ordinal(size_t dim, MultiIndex... inds) {

    // Sort the indices.
    auto ind_vec = std::array<size_t, sizeof...(MultiIndex)>{static_cast<size_t>(inds)...};

    std::sort(ind_vec.begin(), ind_vec.end());

    // Compute the index.
    size_t out = 0;
#pragma unroll
    for(int i = 0; i < sizeof...(MultiIndex); i++) {
        out += choose_up<i>(ind_vec[i]);
    }

    return out;
}

namespace detail {

template<typename T, int n>
int sort_and_perms(std::array<T, n> &arr) {
    int parity = 1; // even parity.

    for(int i = 0; i < n; i++) {
        int best = i;

        // Find the current best element to swap.
        for(int j = i; j < n; j++) {
            if(arr[j] < arr[best]) {
                best = j;
            }
        }

        // Insert.
        std::swap(arr[i], arr[best]);
        parity = -parity; // Swap parity.
    }

    return parity;
}

}

template<typename... MultiIndex>
size_t asymm_index_ordinal(size_t dim, int *sign, MultiIndex... inds) {
    // Sort the indices.
    auto ind_vec = std::array<size_t, sizeof...(MultiIndex)>{static_cast<size_t>(inds)...};

    *sign = detail::sort_and_perms(ind_vec);

    // Compute the index.
    size_t out = 0;
#pragma unroll
    for(int i = 0; i < sizeof...(MultiIndex); i++) {
        out += choose_up<i>(ind_vec[i]);
    }

    return out;
}

END_EINSUMS_NAMESPACE_HPP(einsums)