//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime.hpp>

#include "argparse/argparse.hpp"

namespace einsums {

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

init_Einsums_BufferAllocator::init_Einsums_BufferAllocator() {
    std::perror("Adding functions.\n");
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    einsums::register_arguments(einsums::add_Einsums_BufferAllocator_arguments);
    einsums::register_startup_function(einsums::initialize_Einsums_BufferAllocator);
    einsums::register_shutdown_function(einsums::finalize_Einsums_BufferAllocator);
}

init_Einsums_BufferAllocator detail::initialize_module_Einsums_BufferAllocator;

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments(argparse::ArgumentParser &parser) {
    std::perror("Adding arguments.\n");
    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();

    parser.add_argument("--einsums:buffer-size")
        .default_value("4MB")
        .help("Total size of buffers allocated for tensor contractions.")
        .store_into(global_string["buffer-size"]);

    global_config.attach(detail::Einsums_BufferAllocator_vars::update_max_size);
}

void initialize_Einsums_BufferAllocator() {
    EINSUMS_LOG_TRACE("initializing Einsums/BufferAllocator");
    /// @todo Fill in.
}

void finalize_Einsums_BufferAllocator() {
    EINSUMS_LOG_TRACE("finalizing module Einsums/BufferAllocator");
}

} // namespace einsums