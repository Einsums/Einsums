//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <stdexcept>

namespace einsums {

/**
 * @struct num_argument_error
 *
 * Indicates that a function did not receive the correct amount of arguments.
 */
struct EINSUMS_EXPORT bad_lexical_cast : std::bad_cast {
    using std::bad_cast::bad_cast;
};

} // namespace einsums