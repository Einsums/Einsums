//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/CommandLine.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include "Einsums/CommandLine/CommandLine.hpp"

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
        einsums::register_pre_startup_function(einsums::initialize_Einsums_BufferAllocator);
        einsums::register_shutdown_function(einsums::finalize_Einsums_BufferAllocator);
        is_initialized = true;
    }

    return 0;
}

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments() {
    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();

    static cl::OptionCategory   bufferCategory("Buffer Allocator");
    static cl::Opt<std::string> bufferSize("einsums:buffer-size", {}, "Total size of buffers allocated for tensor contractions",
                                           bufferCategory, cl::Location(global_string["buffer-size"]), cl::Default(std::string("4MB")));
    static cl::Opt<std::string> workBuffersize(
        "einsums:work-buffer-size", {},
        "The largest buffer size to use for buffered contractions. Should be much smaller than the max buffer size. The maximum should be "
        "the value of --einsums:buffer-size divided by three times the number of threads. In reality, the program will need more space for "
        "other buffers, so the size should be much smaller than that. Setting to zero will let the program decide.",
        bufferCategory, cl::Location(global_string["work-buffer-size"]), cl::Default(std::string("0")));

    global_config.attach(detail::Einsums_BufferAllocator_vars::update_max_size);

    auto &singleton = detail::Einsums_BufferAllocator_vars::get_singleton();
    auto  lock      = std::lock_guard(singleton);
    singleton.set_max_size(string_util::memory_string("4MB"));
}

void initialize_Einsums_BufferAllocator() {
    // Create the singleton instance early.
    auto &singleton = detail::Einsums_BufferAllocator_vars::get_singleton();
}

void finalize_Einsums_BufferAllocator() {
}

} // namespace einsums