#define CATCH_CONFIG_RUNNER
#include "einsums/Blas.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Timer.hpp"

#include <catch2/catch_all.hpp>
#include <h5cpp/io>

#if defined(EINSUMS_IN_PARALLEL)
#    include <h5cpp/H5Pall.hpp>
#    include <mpi.h>
#endif

auto main(int argc, char *argv[]) -> int {
    einsums::initialize();

#if defined(EINSUMS_IN_PARALLEL)
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    // Create a file to hold the data from the DiskTensor tests.
    einsums::state::data = h5::create(std::string(std::tmpnam(nullptr)), H5F_ACC_TRUNC, h5::fcpl, h5::mpiio({comm, info}));
#else
    // Create a file to hold the data from the DiskTensor tests.
    einsums::state::data = h5::create(std::string(std::tmpnam(nullptr)), H5F_ACC_TRUNC);
#endif

    // println("Running on {} thread(s)", omp_get_max_threads());

    int result = Catch::Session().run(argc, argv);

    // Ensure file is closed before finalize is called. If einsums is running parallel MPI will have been
    // finalized BEFORE the HDF5 file has been closed and MPI will report that an MPI function was called
    // after having been finalized.
    H5Fclose(einsums::state::data);

    // Passing false means "do not print timer report", whereas passing true will print the
    // timer report.
    einsums::finalize(false);

    return result;
}
