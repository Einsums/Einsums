#include <Einsums/BLAS/InitModule.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>


/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

einsums::init_Einsums_BLAS::init_Einsums_BLAS() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    einsums::detail::global_startup_functions.push_back(einsums::initialize_Einsums_BLAS);
    einsums::detail::global_shutdown_functions.push_back(einsums::initialize_Einsums_BLAS);
}

einsums::init_Einsums_BLAS einsums::detail::initialize_module_Einsums_BLAS;

void einsums::initialize_Einsums_BLAS() {
    blas::vendor::initialize();
}

void einsums::finalize_Einsums_BLAS() {
    blas::vendor::finalize();
}