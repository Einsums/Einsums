//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <tuple>

namespace einsums {
namespace detail {

template <typename T, std::size_t... Is>
constexpr auto create_tuple(T value, std::index_sequence<Is...>) {
    return std::tuple{(static_cast<void>(Is), value)...};
}

template <typename T, std::size_t... Is>
constexpr auto create_tuple_from_array(T const &arr, std::index_sequence<Is...>) {
    return std::tuple((arr[Is])...);
}

} // namespace detail

template <size_t N, typename T>
constexpr auto create_tuple(T const &value) {
    return detail::create_tuple(value, std::make_index_sequence<N>());
}

template <size_t N, typename T>
constexpr auto create_tuple_from_array(T const &arr) {
    return detail::create_tuple_from_array(arr, std::make_index_sequence<N>());
}

} // namespace einsums