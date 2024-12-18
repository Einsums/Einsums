//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstdio>
#include <fstream>
#include <string>

namespace einsums {

/**
 * @brief Handles initializing the internals of Einsums.
 *
 * The current implementation initializes the timer system, calls
 * on the blas subsystem to initialize itself (for example, gpu variant would
 * obtain global device handle), prevents OpenMP from allowing nested
 * OpenMP regions (leading to oversubscription), and disables HDF5
 * diagnostic reporting.
 *
 * In a future parallel variant of Einsums, this would also initialize
 * the MPI runtime.
 *
 * @return int on success returns 0, on failure anything else.
 */
int EINSUMS_EXPORT initialize();

/**
 * Shuts down Einsums and prints out the timing report to the specified file.
 */
EINSUMS_EXPORT void finalize(char const *output_file);
EINSUMS_EXPORT void finalize(std::string const &output_file);
EINSUMS_EXPORT void finalize(FILE *file_pointer);
EINSUMS_EXPORT void finalize(std::ostream &output_stream);

/**
 * Shuts down Einsums and possibly print out a timings report to stdout.
 *
 * @param timerReport whether to print the timings report of not. Defaults to false.
 */
EINSUMS_EXPORT void finalize(bool timerReport = false);

} // namespace einsums