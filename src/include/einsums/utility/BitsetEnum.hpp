#pragma once

#include <type_traits>

namespace einsums {

/**
 * @brief Controls whether an enum class can be used with bitwise operators.
 *
 * enum classes by default are not able to be used with bitwise operators, unlike
 * their plain enum counter parts. Default IsBitsetEnum will evaluate to false.
 *
 * If you wish to use bitwise operators then after you define your enum class hahve
 *
 * @code
 * template<>
 * struct IsBitsetEnum<MyAwesomeEnumClass> : std::true_type {};
 * @endcode
 *
 * @tparam type
 */
template <typename type>
struct IsBitsetEnum : std::false_type {};

template <typename type>
inline constexpr bool IsBitsetEnumV = IsBitsetEnum<type>::value;

/**
 * @brief Concept that controls the bitwise operators for enum classes.
 *
 * @tparam type
 */
template <typename type>
concept BitsetEnum = IsBitsetEnumV<type>;

} // namespace einsums

template <einsums::BitsetEnum type>
auto operator&(const type &lhs, const type &rhs) -> type {
    using Underlying = std::underlying_type_t<type>;
    return static_cast<type>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
}

template <einsums::BitsetEnum type>
auto operator^(const type &lhs, const type &rhs) -> type {
    using Underlying = std::underlying_type_t<type>;
    return static_cast<type>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
}

template <einsums::BitsetEnum type>
auto operator|(const type &lhs, const type &rhs) -> type {
    using Underlying = std::underlying_type_t<type>;
    return static_cast<type>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
}

template <einsums::BitsetEnum type>
auto operator~(const type &lhs) -> type {
    using Underlying = std::underlying_type_t<type>;
    return static_cast<type>(~static_cast<Underlying>(lhs));
}

template <einsums::BitsetEnum type>
auto operator&=(type &lhs, const type &rhs) -> type & {
    using Underlying = std::underlying_type_t<type>;
    lhs              = static_cast<type>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
    return lhs;
}

template <einsums::BitsetEnum type>
auto operator^=(type &lhs, const type &rhs) -> type & {
    using Underlying = std::underlying_type_t<type>;
    lhs              = static_cast<type>(static_cast<Underlying>(lhs) ^ static_cast<Underlying>(rhs));
    return lhs;
}

template <einsums::BitsetEnum type>
auto operator|=(type &lhs, const type &rhs) -> type & {
    using Underlying = std::underlying_type_t<type>;
    lhs              = static_cast<type>(static_cast<Underlying>(lhs) | static_cast<Underlying>(rhs));
    return lhs;
}
