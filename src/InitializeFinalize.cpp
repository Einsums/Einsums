#include "einsums/Blas.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Timer.hpp"
#include "einsums/_Common.hpp"
#include "einsums/parallel/MPI.hpp"

#include <h5cpp/all>

namespace einsums {

auto initialize() -> int {
    ErrorOr<void, mpi::Error> result = mpi::initialize(0, nullptr);
    if (result.is_error())
        return 1;

    timer::initialize();
    blas::initialize();

    // Disable nested omp regions
    omp_set_max_active_levels(1);

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);

    return 0;
}

void finalize(bool timerReport) {
    blas::finalize();

    ErrorOr<void, mpi::Error> result = mpi::finalize();

    if (timerReport)
        timer::report();

    timer::finalize();
}

} // namespace einsums