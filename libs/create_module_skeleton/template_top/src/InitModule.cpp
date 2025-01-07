//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <{lib_name}/{module_name}/InitModule.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Logging.hpp>

namespace einsums {{
    
/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

init_{lib_name}_{module_name}::init_{lib_name}_{module_name}() {{
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    detail::global_startup_functions().emplace_back(initialize_{lib_name}_{module_name});
    detail::global_shutdown_functions().emplace_back(initialize_{lib_name}_{module_name});
}}

init_{lib_name}_{module_name} detail::initialize_module_{lib_name}_{module_name};

void initialize_{lib_name}_{module_name}() {{
    EINSUMS_LOG_TRACE("initializing module");
    // TODO: Fill in.
}}

void finalize_{lib_name}_{module_name}() {{
    EINSUMS_LOG_TRACE("finalizing module");
    // TODO: Fill in.
}}

}}