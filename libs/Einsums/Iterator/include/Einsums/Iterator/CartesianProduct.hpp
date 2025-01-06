//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TypeSupport/Tuple.hpp>

#include <array>
#include <tuple>
#include <vector>

namespace einsums {
namespace detail {

// Utility to calculate product of container sizes.
template <typename Tuple>
constexpr std::size_t tuple_product(Tuple const &tuple) {
    return std::apply([](auto... sizes) { return (sizes * ... * 1); }, tuple);
}

// Helper to get size of a container
template <typename T>
constexpr std::size_t container_size(T const &container) {
    return container.size();
}

// Specialization for std::tuple
template <typename... Args>
constexpr std::size_t container_size(std::tuple<Args...> const &) {
    return sizeof...(Args);
}

// Helper to access elements in a container
template <typename T>
constexpr auto container_access(T const &container, std::size_t index) {
    return container[index];
}

// Specialization for std::tuple
template <typename... Args>
constexpr auto container_access(std::tuple<Args...> const &container, std::size_t index) {
    return std::apply([&](auto &&...args) { return std::array{std::forward<decltype(args)>(args)...}[index]; }, container);
}

} // namespace detail

template <typename... Containers>
struct CartesianProduct {
    using IndexTuple = decltype(create_tuple<sizeof...(Containers)>());

    CartesianProduct(Containers const &...containers)
        : _containers(std::tie(containers...)), _sizes{detail::container_size(containers)...}, _total_size{detail::tuple_product(_sizes)} {}

    struct Iterator {
        Iterator(CartesianProduct const *parent, std::size_t index) : _parent(parent), _index(index) {}

        bool operator!=(Iterator const &other) const { return _index != other._index; }
        bool operator==(Iterator const &other) const { return _index == other._index; }

        Iterator &operator++() {
            ++_index;
            return *this;
        }

        Iterator  operator+(std::size_t n) const { return Iterator(_parent, _index + n); }
        Iterator &operator+=(std::size_t n) {
            _index += n;
            return *this;
        }

        auto operator*() const {
            IndexTuple indices = calculate_indices(_index);
            return dereference(indices, std::make_index_sequence<sizeof...(Containers)>{});
        }

        std::ptrdiff_t operator-(Iterator const &other) const {
            return static_cast<std::ptrdiff_t>(_index) - static_cast<std::ptrdiff_t>(other._index);
        }

      private:
        CartesianProduct const *_parent;
        std::size_t             _index;

        IndexTuple calculate_indices(std::size_t flat_index) const {
            IndexTuple  indices{};
            std::size_t divisor = _parent->_total_size;

            // Iterate over all elements in parent->sizes using pack expansion
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                ((divisor /= std::get<Is>(_parent->_sizes), std::get<Is>(indices) = flat_index / divisor, flat_index %= divisor), ...);
            }(std::make_index_sequence<sizeof...(Containers)>{});

            return indices;
        }

        template <std::size_t... Is>
        auto dereference(IndexTuple const &indices, std::index_sequence<Is...>) const {
            return std::make_tuple(detail::container_access(std::get<Is>(_parent->_containers), std::get<Is>(indices))...);
        }
    };

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, _total_size); }

  private:
    std::tuple<Containers const &...> _containers;
    IndexTuple                        _sizes;
    std::size_t                       _total_size;
};

} // namespace einsums