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

void omp_set_num_threads(int nthread) {
    (void)nthread;
}

int omp_in_parallel() {
    return 0;
}