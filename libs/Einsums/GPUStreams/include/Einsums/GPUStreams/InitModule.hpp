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
 * @class init_Einsums_GPUStreams
 *
 * Auto-generated class. The constructor registers the initialization and finalization functions.
 */
class EINSUMS_EXPORT init_Einsums_GPUStreams {
public:
    init_Einsums_GPUStreams();
};

EINSUMS_HOST EINSUMS_EXPORT void initialize_Einsums_GPUStreams();
EINSUMS_HOST EINSUMS_EXPORT void finalize_Einsums_GPUStreams();

namespace detail {

extern EINSUMS_EXPORT init_Einsums_GPUStreams initialize_module_Einsums_GPUStreams;

}

}