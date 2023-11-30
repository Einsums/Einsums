//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"

#include <h5cpp/core>

// Items in this namespace contribute to the global state of the program
namespace einsums::state {

extern h5::fd_t EINSUMS_EXPORT data;
extern h5::fd_t EINSUMS_EXPORT checkpoint_file;

} // namespace einsums::state