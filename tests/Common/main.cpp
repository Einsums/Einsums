#define CATCH_CONFIG_RUNNER
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/State.hpp"
#include "einsums/Timer.hpp"

#include <catch2/catch.hpp>
#include <h5cpp/io>

auto main(int argc, char *argv[]) -> int {
    einsums::timer::initialize();

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);

    // Create a file to hold the data from the DiskTensor tests.
    einsums::state::data = h5::create("Data.h5", H5F_ACC_TRUNC);

    println("Running on {} thread(s)", omp_get_max_threads());

    int result = Catch::Session().run(argc, argv);

    einsums::timer::report();
    einsums::timer::finalize();
    return result;
}