//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>

namespace einsums {

struct ErrorCode;
struct EINSUMS_EXPORT Exception;

enum class ThrowMode : std::uint8_t { plain = 0, rethrow = 1 };

EINSUMS_EXPORT extern ErrorCode throws;

} // namespace einsums