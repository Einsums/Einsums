//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <type_traits>
namespace einsums {

template <typename T>
struct SizeOf {
    constexpr static size_t value = sizeof(T);
};

template <>
struct SizeOf<void> {
    constexpr static size_t value = 1;
};

template <>
struct SizeOf<void const> {
    constexpr static size_t value = 1;
};

/**
 * @property SizeOf
 *
 * @brief Provides extended @c sizeof functionality.
 *
 * Because @c sizeof(void) is an invalid statement, this gives a value for it. In particular,
 * since @c void is used for typeless data, we generally want it to have a size of one byte,
 * especially for anything that deals with generic memory manipulations.
 *
 * @tparam T The type to measure.
 */
template <typename T>
constexpr size_t SizeOfV = SizeOf<T>::value;
} // namespace einsums