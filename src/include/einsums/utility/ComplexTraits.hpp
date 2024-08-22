//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/utility/TensorTraits.hpp"

#include <complex>
#include <type_traits>

namespace einsums {

template <typename>
inline constexpr bool IsComplexV = false;

template <typename type>
    requires(!TensorConcept<type>)
inline constexpr bool IsComplexV<std::complex<type>> =
    std::disjunction_v<std::is_same<type, float>, std::is_same<type, double>, std::is_same<type, long double>>;

template <TensorConcept TensorType>
inline constexpr bool IsComplexV<TensorType> = IsComplexV<typename TensorType::data_type>;

template <typename type>
struct IsComplex : std::bool_constant<IsComplexV<type>> {};

template <typename type>
concept Complex = IsComplex<type>::value;

template <typename type>
concept NotComplex = !IsComplex<type>::value;

template <typename T>
struct ComplexType {
    using Type = T;
};

template <TensorConcept T>
struct ComplexType<T> {
    using Type = typename ComplexType<typename T::data_type>::Type;
};

template <typename T>
struct ComplexType<std::complex<T>> {
    using Type = T;
};

template <typename T>
using RemoveComplexT = typename ComplexType<T>::Type;

template <typename T>
struct AddComplex {
    using Type = std::complex<T>;
};

template <typename T>
struct AddComplex<std::complex<T>> {
    using Type = std::complex<T>;
};

template <typename T>
using AddComplexT = typename AddComplex<T>::Type;

} // namespace einsums