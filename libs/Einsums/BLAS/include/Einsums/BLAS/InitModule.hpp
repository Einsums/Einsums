//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 */

namespace einsums {

EINSUMS_EXPORT int setup_Einsums_BLAS();

/**
 * @brief Initialize the BLAS runtime.
 */
EINSUMS_EXPORT void initialize_Einsums_BLAS();

/**
 * @brief Finalize the BLAS runtime.
 */
EINSUMS_EXPORT void finalize_Einsums_BLAS();

namespace detail {

static int initialize_module_Einsums_BLAS = setup_Einsums_BLAS();

}

} // namespace einsums