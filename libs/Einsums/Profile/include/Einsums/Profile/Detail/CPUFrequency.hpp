//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstdint>

namespace einsums::profile::detail {

/// @brief Return the CPU frequency in Hz.
EINSUMS_EXPORT auto cpu_frequency() -> uint64_t;

}