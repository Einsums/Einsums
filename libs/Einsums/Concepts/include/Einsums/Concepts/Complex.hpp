//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Tensor.hpp>

#include <complex>
#include <type_traits>

namespace einsums {

template <typename>
inline constexpr bool IsComplexV = false;

template <typename T>
inline constexpr bool IsComplexV<std::complex<T>> = std::is_floating_point_v<T>;

template <typename T>
concept IsComplex = requires(T t) { requires std::same_as<T, std::complex<typename T::value_type>>; };

template <typename T>
concept IsComplexTensor = requires(T t) {
    requires std::same_as<typename T::value_type, std::complex<typename T::value_type::value_type>>;
    requires TensorConcept<T>;
};

template <typename T>
concept Complex = IsComplex<T> || IsComplexTensor<T>;

template <typename T>
concept NotComplex = !Complex<T>;

namespace detail {
template <typename T>
struct RemoveComplex {
    using type = T;
};

template <typename T>
struct RemoveComplex<std::complex<T>> {
    using type = T;
};
} // namespace detail

template <typename T>
using RemoveComplexT = typename detail::RemoveComplex<T>::type;

namespace detail {
template <typename T>
struct AddComplex {
    using type = std::complex<T>;
};

template <typename T>
struct AddComplex<std::complex<T>> {
    using type = std::complex<T>;
};
} // namespace detail

template <typename T>
using AddComplexT = typename detail::AddComplex<T>::type;

} // namespace einsums