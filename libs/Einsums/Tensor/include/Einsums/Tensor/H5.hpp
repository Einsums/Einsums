//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Tensor/TensorForward.hpp>

#include <array>
#include <h5cpp/core>
#include <h5cpp/io>
#include <type_traits>

namespace h5 {
namespace impl {

#ifndef DOXYGEN
template <typename T, size_t Rank>
struct decay<::einsums::Tensor<T, Rank>> {
    using type = T;
};

template <typename T, size_t Rank>
struct decay<::einsums::TensorView<T, Rank>> {
    using type = T;
};
#endif

/**
 * Gets the pointer to the data contained in a tensor.
 *
 * @param ref The tensor to query.
 */
template <typename T, size_t Rank>
auto data(::einsums::Tensor<T, Rank> const &ref) -> T const * {
    return ref.data();
}

/**
 * Gets the pointer to the data contained in a tensor view.
 *
 * @param ref The tensor to query.
 */
template <typename T, size_t Rank>
auto data(::einsums::TensorView<T, Rank> const &ref) -> T const * {
    return ref.data();
}

/**
 * Gets the pointer to the data contained in a tensor.
 *
 * @param ref The tensor to query.
 */
template <typename T, size_t Rank>
auto data(::einsums::Tensor<T, Rank> &ref) -> T * {
    return ref.data();
}

/**
 * @brief Determines the rank of a tensor or view.
 *
 */
template <typename T, size_t Rank>
struct rank<::einsums::Tensor<T, Rank>> : public std::integral_constant<size_t, Rank> {};

template <typename T, size_t Rank>
struct rank<::einsums::TensorView<T, Rank>> : public std::integral_constant<size_t, Rank> {};

/**
 * @brief Determines the dimensions of a tensor or view.
 *
 */
template <typename T, size_t Rank>
inline auto size(::einsums::Tensor<T, Rank> const &ref) -> std::array<std::int64_t, Rank> {
    return ref.dims();
}

template <typename T, size_t Rank>
inline auto size(::einsums::TensorView<T, Rank> const &ref) -> std::array<std::int64_t, Rank> {
    return ref.dims();
}

#ifndef DOXYGEN
template <typename T, size_t Rank>
struct get<::einsums::Tensor<T, Rank>> {
    static inline auto ctor(std::array<size_t, Rank> dims) -> ::einsums::Tensor<T, Rank> {
        auto ctor_bind = std::bind_front(::einsums::Tensor<T, Rank>::Tensor, "hdf5 auto created");
        return std::apply(ctor_bind, dims);
    }
};
#endif

} // namespace impl

inline bool exists(hid_t hid, std::string const &name) {
    return H5Lexists(hid, name.c_str(), H5P_DEFAULT) > 0 ? true : false;
}

} // namespace h5