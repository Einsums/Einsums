#include <Einsums/Utilities/InitModule.hpp>
#include <Einsums/Utilities/Random.hpp>
#include <Einsums/Runtime.hpp>
#include <random>

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

einsums::init_Einsums_Utilities::init_Einsums_Utilities() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    einsums::detail::global_startup_functions.push_back(einsums::initialize_Einsums_Utilities);
    einsums::detail::global_shutdown_functions.push_back(einsums::initialize_Einsums_Utilities);
}

einsums::init_Einsums_Utilities einsums::detail::initialize_module_Einsums_Utilities;

void einsums::initialize_Einsums_Utilities() {
    einsums::random_engine = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
}

void einsums::finalize_Einsums_Utilities() {
    // TODO: Fill in.
}