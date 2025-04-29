
//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>

namespace einsums {

template <typename T>
concept Enum = std::is_enum_v<T>;

template <typename T>
concept ScopedEnum = std::is_enum_v<T> && !std::is_convertible_v<T, int>;

} // namespace einsums