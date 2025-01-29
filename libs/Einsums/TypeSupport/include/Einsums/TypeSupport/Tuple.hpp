//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <tuple>

namespace einsums {
namespace detail {

template <typename T, std::size_t... Is>
constexpr auto create_tuple(std::index_sequence<Is...>) {
    return std::tuple((static_cast<void>(Is), T{})...);
}

} // namespace detail

template <std::size_t N, typename T = std::size_t>
constexpr auto create_tuple() {
    return detail::create_tuple<T>(std::make_index_sequence<N>());
}

} // namespace einsums
