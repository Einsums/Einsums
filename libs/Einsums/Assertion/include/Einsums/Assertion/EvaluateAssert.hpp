//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <source_location>
#include <string>

namespace einsums::detail {

/**
 * @brief Handles assertion behavior, selecting between a user-defined handler or the default handler.
 *
 * @param loc The source location for constructing a message.
 * @param expr A string representing the condition being evaluated.
 * @param msg An extra message to print out.
 */
EINSUMS_EXPORT void handle_assert(std::source_location const &loc, char const *expr, std::string const &msg) noexcept;

} // namespace einsums::detail