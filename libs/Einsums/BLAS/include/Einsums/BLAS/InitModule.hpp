#pragma once

#include <Einsums/Config.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 */

namespace einsums {

/**
 * @class init_Einsums_BLAS
 *
 * Auto-generated class. The constructor registers the initialization and finalization functions.
 */
class EINSUMS_EXPORT init_Einsums_BLAS {
public:
    init_Einsums_BLAS();
};

/**
 * @brief Initialize the BLAS runtime.
 */
EINSUMS_EXPORT void initialize_Einsums_BLAS();

/**
 * @brief Finalize the BLAS runtime.
 */
EINSUMS_EXPORT void finalize_Einsums_BLAS();

namespace detail {

extern EINSUMS_EXPORT init_Einsums_BLAS initialize_module_Einsums_BLAS;

}

}