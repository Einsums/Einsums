//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>
#include "argparse/argparse.hpp"

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 */

namespace einsums {

/**
 * @class init_Einsums_BufferAllocator
 *
 * Auto-generated class. The constructor registers the initialization and finalization functions.
 */
class EINSUMS_EXPORT init_Einsums_BufferAllocator {
public:
    init_Einsums_BufferAllocator();
};

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments(argparse::ArgumentParser &);
EINSUMS_EXPORT void initialize_Einsums_BufferAllocator();
EINSUMS_EXPORT void finalize_Einsums_BufferAllocator();

namespace detail {

extern EINSUMS_EXPORT init_Einsums_BufferAllocator initialize_module_Einsums_BufferAllocator;

}

}