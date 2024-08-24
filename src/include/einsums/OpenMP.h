//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(_OPENMP)
#    include <omp.h>
#else

#    include "einsums/_Export.hpp"

#    if defined(__cplusplus)
extern "C" {
#    endif

int EINSUMS_EXPORT  omp_get_max_threads();
int EINSUMS_EXPORT  omp_get_num_threads();
void EINSUMS_EXPORT omp_set_num_threads(int);
int EINSUMS_EXPORT  omp_get_thread_num();
int EINSUMS_EXPORT  omp_in_parallel();

/**
 * @brief A nonzero value enables nested parallelism, while zero disables nested parallelism.
 *
 * @param val
 */
void EINSUMS_EXPORT omp_set_nested(int val);

/**
 * @brief A nonzero value means nested parallelism is enabled.
 *
 * @return int
 */
int EINSUMS_EXPORT omp_get_nested();

void EINSUMS_EXPORT omp_set_max_active_levels(int max_levels);

#    if defined(__cplusplus)
}
#    endif

#endif