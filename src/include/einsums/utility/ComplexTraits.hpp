//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/utility/TensorTraits.hpp"

#include <complex>
#include <type_traits>

namespace einsums {

/**
 * @property IsComplexV
 *
 * @brief Checks if the type holds a complex value.
 *
 * Evaluates to true if the type provided is a complex type, or if the type is a tensor storing complex values.
 */
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

/**
 * @concept Complex
 *
 * @brief Wraps IsComplexV.
 */
template <typename type>
concept Complex = IsComplex<type>::value;

/**
 * @concept NotComplex
 *
 * @brief The opposite of Complex.
 */
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

/**
 * @typedef RemoveComplexT
 *
 * @brief If the provided type is complex, return the real version.
 *
 * If the provided type is complex, this gets the underlying real type. If the type
 * is real, then this returns the type. If the type is a tensor, it passes its held
 * type through this typedef and retrieves its real type.
 */
template <typename T>
using RemoveComplexT = typename ComplexType<T>::Type;

template <typename T>
struct AddComplex {
    using Type = std::complex<T>;
};

template <TensorConcept T>
struct AddComplex<T> {
    using Type = typename AddComplex<typename T::data_type>::Type;
};

template <typename T>
struct AddComplex<std::complex<T>> {
    using Type = std::complex<T>;
};

/**
 * @typedef AddComplexT
 *
 * @brief Creates a complex type from the provided type.
 *
 * If AddComplexT is provided a complex type, it will do nothing.
 * If it is passed a real type, then it will return the complex type that holds that real type.
 * If it is passed a tensor, then this will apply its transformation on the type held by that tensor.
 */
template <typename T>
using AddComplexT = typename AddComplex<T>::Type;

} // namespace einsums