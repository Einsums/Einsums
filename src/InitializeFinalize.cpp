#include "einsums/Blas.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Timer.hpp"
#include "einsums/_Common.hpp"

namespace einsums {

auto initialize() -> int {
    timer::initialize();
    blas::initialize();

    // Disable nested omp regions
    omp_set_max_active_levels(1);

    return 0;
}

void finalize(bool timerReport) {
    blas::finalize();

    if (timerReport)
        timer::report();

    timer::finalize();
}

} // namespace einsums