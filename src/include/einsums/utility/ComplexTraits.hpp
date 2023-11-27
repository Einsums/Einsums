#pragma once

#include <complex>
#include <type_traits>

namespace einsums {

template <typename>
inline constexpr bool IsComplexV = false;

template <typename type>
inline constexpr bool IsComplexV<std::complex<type>> =
    std::disjunction_v<std::is_same<type, float>, std::is_same<type, double>, std::is_same<type, long double>>;

template <typename type>
struct IsComplex : std::bool_constant<IsComplexV<type>> {};

template <typename type>
concept Complex = IsComplexV<type>;

template <typename T>
struct ComplexType {
    using Type = T;
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