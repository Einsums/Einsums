#pragma once

#include <Einsums/Config.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 */

namespace einsums {

EINSUMS_EXPORT int setup_Einsums_GPUStreams();

/**
 * @brief Set up the GPU, as well as the various streams for threading.
 */
EINSUMS_HOST EINSUMS_EXPORT void initialize_Einsums_GPUStreams();

/**
 * @brief Free data related to the GPU.
 */
EINSUMS_HOST EINSUMS_EXPORT void finalize_Einsums_GPUStreams();

namespace detail {

static int initialize_module_Einsums_GPUStreams = setup_Einsums_GPUStreams();

}

} // namespace einsums