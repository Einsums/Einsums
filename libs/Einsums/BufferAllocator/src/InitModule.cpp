//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime.hpp>

#include <argparse/argparse.hpp>

namespace einsums {

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

int init_Einsums_BufferAllocator() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_arguments(einsums::add_Einsums_BufferAllocator_arguments);
        einsums::register_startup_function(einsums::initialize_Einsums_BufferAllocator);
        einsums::register_shutdown_function(einsums::finalize_Einsums_BufferAllocator);
        is_initialized = true;
    }

    return 0;
}

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments(argparse::ArgumentParser &parser) {
    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();

    parser.add_argument("--einsums:buffer-size")
        .default_value("4MB")
        .help("Total size of buffers allocated for tensor contractions.")
        .store_into(global_string["buffer-size"]);

    global_config.attach(detail::Einsums_BufferAllocator_vars::update_max_size);
}

void initialize_Einsums_BufferAllocator() {
    /// @todo Fill in.
}

void finalize_Einsums_BufferAllocator() {
}

} // namespace einsums