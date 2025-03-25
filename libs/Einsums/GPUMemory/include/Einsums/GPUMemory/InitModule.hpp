//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>
#include <argparse/argparse.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 */

namespace einsums {

EINSUMS_EXPORT int init_Einsums_GPUMemory();

EINSUMS_EXPORT void add_Einsums_GPUMemory_arguments(argparse::ArgumentParser &);
EINSUMS_EXPORT void initialize_Einsums_GPUMemory();
EINSUMS_EXPORT void finalize_Einsums_GPUMemory();

namespace detail {

static int initialize_module_Einsums_GPUMemory = init_Einsums_GPUMemory();

}

}