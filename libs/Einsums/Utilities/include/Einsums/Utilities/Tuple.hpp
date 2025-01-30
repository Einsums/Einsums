//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

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

template <size_t NewN, typename T, size_t OldN, size_t... I>
constexpr inline std::array<T, NewN> slice_array_impl(std::array<T, OldN> const &old_array, std::index_sequence<I...> const &) {
    return {{old_array[I]...}};
}

template <size_t NewN, typename T, size_t OldN, size_t... I>
constexpr inline std::array<T, NewN> slice_array_impl(std::array<T, OldN> &&old_array, std::index_sequence<I...> const &) {
    return {{std::move(old_array[I])...}};
}

template <typename T, typename... TupleTypes, size_t... I>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple_impl(std::tuple<TupleTypes...> const &tuple,
                                                                                   std::index_sequence<I...> const &) {
    return {{std::get<I>(tuple)...}};
}

template <typename T, typename... TupleTypes, size_t... I>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple_impl(std::tuple<TupleTypes...> &&tuple,
                                                                                   std::index_sequence<I...> const &) {
    return {{std::move(std::get<I>(tuple))...}};
}

template <typename NewT, typename OldT, size_t N, size_t... I>
constexpr inline std::array<std::remove_cv_t<NewT>, N> convert_array_impl(std::array<std::remove_cv_t<OldT>, N> const &arr,
                                                                          std::index_sequence<I...> const &) {
    return {{static_cast<std::remove_cv_t<NewT>>(arr[I])...}};
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

template <size_t NewN, typename T, size_t OldN>
    requires(OldN >= NewN)
constexpr std::array<T, NewN> slice_array(std::array<T, OldN> const &old_array) {
    return detail::slice_array_impl<NewN>(old_array, std::make_index_sequence<NewN>());
}

template <size_t NewN, typename T, size_t OldN>
    requires(OldN >= NewN)
constexpr std::array<T, NewN> slice_array(std::array<T, OldN> &&old_array) {
    return detail::slice_array_impl<NewN>(std::move(old_array), std::make_index_sequence<NewN>());
}

template <typename T, typename... TupleTypes>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple(std::tuple<TupleTypes...> const &tuple) {
    return detail::create_array_from_tuple_impl<T>(tuple, std::make_index_sequence<sizeof...(TupleTypes)>());
}

template <typename T, typename... TupleTypes>
    requires(std::is_convertible_v<TupleTypes, T> && ...)
constexpr inline std::array<T, sizeof...(TupleTypes)> create_array_from_tuple(std::tuple<TupleTypes...> &&tuple) {
    return detail::create_array_from_tuple_impl<T>(std::move(tuple), std::make_index_sequence<sizeof...(TupleTypes)>());
}

template <typename NewT, typename OldT, size_t N>
    requires(std::is_convertible_v<OldT, NewT>)
constexpr inline std::array<std::remove_cv_t<NewT>, N> convert_array(std::array<std::remove_cv_t<OldT>, N> const &arr) {
    return detail::convert_array_impl(arr, std::make_index_sequence<N>());
}

} // namespace einsums