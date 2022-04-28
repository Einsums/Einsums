#include "einsums/OpenMP.h"

int omp_get_max_threads() {
    return 1;
}

int omp_get_thread_num() {
    return 0;
}

int omp_get_num_threads() {
    return 1;
}
