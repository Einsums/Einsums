//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"

#include <h5cpp/core>

// Items in this namespace contribute to the global state of the program
namespace einsums::state {

EINSUMS_EXPORT h5::fd_t& data();
EINSUMS_EXPORT h5::fd_t& checkpoint_file();

} // namespace einsums::state