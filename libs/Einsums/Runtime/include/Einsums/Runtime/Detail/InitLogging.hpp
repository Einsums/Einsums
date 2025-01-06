//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

namespace einsums::detail {
EINSUMS_EXPORT void init_logging(RuntimeConfiguration &config);
}