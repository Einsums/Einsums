#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Index.hpp"

#include <algorithm>
#include <cstdarg>
#include <numeric>
#include <queue>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::symm_index)

namespace detail {

EINSUMS_EXPORT size_t calc_index(size_t n, size_t k);

}

template <typename... Idx>
struct SymmIndex {
  private:
    std::tuple<Idx...> _inds;

  public:
    SymmIndex() = default;

    size_t operator()(std::va_list inds) const {
        auto ords = std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>>();

        for_sequence<sizeof...(Idx)>([this, inds, &ords](auto i) { ords.push(std::get<i>(_inds)(inds)); });

        size_t out = 0;

        for (int i = 0; i < sizeof...(Idx); i++) {
            size_t curr = ords.top();
            ords.pop();
            out += detail::calc_index(curr, i + 1);
        }

        return out;
    }

    template<typename IndContainer>
    size_t operator()(IndContainer *inds) const {
        auto ords = std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>>();

        for_sequence<sizeof...(Idx)>([this, inds, &ords](auto i) { ords.push(std::get<i>(_inds)(inds)); });

        size_t out = 0;

        for (int i = 0; i < sizeof...(Idx); i++) {
            size_t curr = ords.top();
            ords.pop();
            out += detail::calc_index(curr, i + 1);
        }

        return out;
    }
};

template <typename... Idx>
struct IndexList {
    private:
    std::tuple<Idx...> _inds;
  public:
    IndexList() = default;

    IndexList(const IndexList<Idx...> &) = default;

    size_t operator()(const Stride<sizeof...(Idx)> &strides, size_t inds...) const {
        std::va_list args;

        va_start(args, inds);

        auto ords = std::array<size_t, sizeof...(Idx)>();

        for_sequence<sizeof...(Idx)>([this, &ords, &args](auto i) { ords[i] = std::get<i>(_inds)(args); });

        size_t out = 0;

        for (int i = 0; i < sizeof...(Idx); i++) {
            out += ords[i] * strides[i];
        }

        va_end(args);

        return out;
    }

    template<typename IndContainer>
    size_t operator()(const Stride<sizeof...(Idx)> &strides, IndContainer *inds) const {
        auto ords = std::array<size_t, sizeof...(Idx)>();

        for_sequence<sizeof...(Idx)>([this, &ords, inds](auto i) { ords[i] = std::get<i>(_inds)(inds); });

        size_t out = 0;

        for (int i = 0; i < sizeof...(Idx); i++) {
            out += ords[i] * strides[i];
        }

        return out;
    }

    template<size_t Rank>
    Dim<sizeof...(Idx)> find_true_dims(Dim<Rank> dims) const {
        std::vector<size_t> dim_vector(Rank);

        for(int i = 0; i < Rank; i++) {
            dim_vector[i] = dims[i] - 1;
        }

        auto ords = Dim<sizeof...(Idx)>();

        for_sequence<sizeof...(Idx)>([this, &ords, &dim_vector](auto i) { ords[i] = std::get<i>(_inds)(&dim_vector); });

        return ords;
    }
};

END_EINSUMS_NAMESPACE_HPP(einsums::symm_index)