#pragma once

#if defined(_OPENMP)
#include <omp.h>
#else

#if defined(__cplusplus)
extern "C" {
#endif

int omp_get_max_threads();

#if defined(__cplusplus)
}
#endif

#endif