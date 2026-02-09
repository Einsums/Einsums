//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Alias.hpp>
#include <Einsums/Config/BranchHints.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/Debug.hpp>
#include <Einsums/Config/Defines.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Types.hpp>
#include <Einsums/Config/Version.hpp>

/**
 * @def EINSUMS_ZERO
 *
 * @brief A macro for indicating small values.
 *
 * This macro is used to indicate when a value is close enough to zero to consider it to be zero.
 *
 * @versionadded{1.0.0}
 */
#if !defined(EINSUMS_ZERO)
#    define EINSUMS_ZERO (1.0e-10)
#endif
