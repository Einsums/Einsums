//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUMemory/InitModule.hpp>
#include <Einsums/GPUMemory/ModuleVars.hpp>
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

int init_Einsums_GPUMemory() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_arguments(einsums::add_Einsums_GPUMemory_arguments);
        einsums::register_startup_function(einsums::initialize_Einsums_GPUMemory);
        einsums::register_shutdown_function(einsums::finalize_Einsums_GPUMemory);
        is_initialized = true;
    }

    return 0;
}

void add_Einsums_GPUMemory_arguments() {
    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();

    static cl::OptionCategory   GPUAllocatorCategory("GPU Buffer Allocator");
    static cl::Opt<std::string> bufferSize("einsums:gpu-buffer-size", {}, "Total size of GPU buffers allocated for tensor contractions.",
                                           GPUAllocatorCategory, cl::Location(global_string["gpu-buffer-size"]),
                                           cl::Default(std::string("4MB")));

    global_config.attach(gpu::detail::Einsums_GPUMemory_vars::update_max_size);
}

void initialize_Einsums_GPUMemory() {
    EINSUMS_LOG_TRACE("initializing Einsums/GPUMemory");

    gpu::detail::Einsums_GPUMemory_vars::get_singleton().reset_curr_size();
}

void finalize_Einsums_GPUMemory() {
    EINSUMS_LOG_TRACE("finalizing module Einsums/GPUMemory");
}

} // namespace einsums