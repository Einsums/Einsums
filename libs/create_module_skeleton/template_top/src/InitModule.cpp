#include <{lib_name}/{module_name}/InitModule.hpp>
#include <Einsums/Runtime.hpp>

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

einsums::init_{lib_name}_{module_name}::init_{lib_name}_{module_name}() {{
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    einsums::register_startup_function(einsums::initialize_{lib_name}_{module_name});
    einsums::register_shutdown_function(einsums::finalize_{lib_name}_{module_name});
}}

einsums::init_{lib_name}_{module_name} einsums::detail::initialize_module_{lib_name}_{module_name};

void einsums::initialize_{lib_name}_{module_name}() {{
    // TODO: Fill in.
}}

void einsums::finalize_{lib_name}_{module_name}() {{
    // TODO: Fill in.
}}