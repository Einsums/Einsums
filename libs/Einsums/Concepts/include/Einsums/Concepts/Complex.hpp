//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>

#include <complex>
#include <type_traits>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {

/**
 * @var IsComplexV
 *
 * @brief Tests whether a type is a complex type.
 *
 * @tparam T The type to test.
 */
template <typename>
inline constexpr bool IsComplexV = false;

/**
 * @copydoc IsComplexV
 */
template <typename T>
inline constexpr bool IsComplexV<std::complex<T>> = std::is_arithmetic_v<T>;

#ifdef EINSUMS_COMPUTE_CODE
template <>
inline constexpr bool IsComplexV<hipFloatComplex> = true;
template <>
inline constexpr bool IsComplexV<hipDoubleComplex> = true;
#endif

/**
 * @concept IsComplex
 *
 * @brief Tests whether a type is a complex type.
 *
 * @tparam T The type to test.
 */
template <typename T>
concept IsComplex = IsComplexV<T>;

/**
 * @concept CanBeComplex
 *
 * @brief Tests whether a type is an arithmetic type including complex.
 *
 * @tparam T The type to test.
 */
template <typename T>
concept CanBeComplex = IsComplexV<T> || !IsComplexV<T>;

/**
 * @concept IsComplexTensor
 *
 * @brief Tests whether a tensor stores a complex data type.
 *
 * @tparam T The type to test.
 */
template <typename T>
concept IsComplexTensor = requires {
    requires IsComplex<typename T::ValueType>;
    requires TensorConcept<T>;
};

/**
 * @concept Complex
 *
 * @brief Tests whether a type is complex, or is a tensor storing a complex value.
 *
 * @tparam T The type to test.
 */
template <typename T>
concept Complex = IsComplex<T> || IsComplexTensor<T>;

/**
 * @concept NotComplex
 *
 * @brief The opposite of Complex.
 *
 * @tparam T The type to test.
 */
template <typename T>
concept NotComplex = !Complex<T>;

namespace detail {

/**
 * @struct RemoveComplex
 *
 * @brief If given a complex type, it will give the real equivalent. If given a real type, no change will happen.
 *
 * @tparam T The type to modify.
 */
template <typename T>
struct RemoveComplex {
    using type = T;
};

template <typename T>
struct RemoveComplex<std::complex<T>> {
    using type = T;
};
} // namespace detail

/**
 * @typedef RemoveComplexT
 *
 * @brief If given a complex type, it will give the real equivalent. If given a real type, no change will happen.
 *
 * @tparam T The type to modify.
 */
template <typename T>
using RemoveComplexT = typename detail::RemoveComplex<T>::type;

namespace detail {

/**
 * @struct AddComplex
 *
 * @brief If given a real type, it will give the complex equivalent. If given a complex type, no change will happen.
 *
 * @tparam T The type to modify.
 */
template <typename T>
struct AddComplex {
    using type = std::complex<T>;
};

template <typename T>
struct AddComplex<std::complex<T>> {
    using type = std::complex<T>;
};
} // namespace detail

/**
 * @typedef AddComplexT
 *
 * @brief If given a real type, it will give the complex equivalent. If given a complex type, no change will happen.
 *
 * @tparam T The type to modify.
 */
template <typename T>
using AddComplexT = typename detail::AddComplex<T>::type;

} // namespace einsums