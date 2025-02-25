//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>

namespace einsums {

/**
 * @brief Finds the number of items with the specified type.
 *
 * @tparam T The type to find.
 * @tparam Args The types to compare against.
 *
 * @return The number of Args that match the T type.
 */
template <typename T, typename... Args>
constexpr auto count_of_type(/*Args... args*/) {
    return (std::is_convertible_v<Args, T> + ... + 0);
}

} // namespace einsums