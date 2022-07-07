#pragma once

#if defined(_OPENMP)
#include <omp.h>
#else

#if defined(__cplusplus)
extern "C" {
#endif

int omp_get_max_threads();
int omp_get_num_threads();
void omp_set_num_threads(int);
int omp_get_thread_num();
int omp_in_parallel();

#if defined(__cplusplus)
}
#endif

#endif