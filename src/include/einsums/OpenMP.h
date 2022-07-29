#pragma once

#if defined(_OPENMP)
#include <omp.h>
#else

#include "einsums/_Export.hpp"

#if defined(__cplusplus)
extern "C" {
#endif

int EINSUMS_EXPORT omp_get_max_threads();
int EINSUMS_EXPORT omp_get_num_threads();
void EINSUMS_EXPORT omp_set_num_threads(int);
int EINSUMS_EXPORT omp_get_thread_num();
int EINSUMS_EXPORT omp_in_parallel();

#if defined(__cplusplus)
}
#endif

#endif