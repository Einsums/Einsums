//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/DesignPatterns/InitModule.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Logging.hpp>

namespace einsums {
    
/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

init_Einsums_DesignPatterns::init_Einsums_DesignPatterns() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    einsums::register_startup_function(einsums::initialize_Einsums_DesignPatterns);
    einsums::register_shutdown_function(einsums::finalize_Einsums_DesignPatterns);
}

init_Einsums_DesignPatterns detail::initialize_module_Einsums_DesignPatterns;

void initialize_Einsums_DesignPatterns() {
    EINSUMS_LOG_TRACE("initializing module");
    // TODO: Fill in.
}

void finalize_Einsums_DesignPatterns() {
    EINSUMS_LOG_TRACE("finalizing module");
    // TODO: Fill in.
}

}