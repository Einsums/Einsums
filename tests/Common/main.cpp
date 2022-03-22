#define CATCH_CONFIG_RUNNER
#include "EinsumsInCpp/OpenMP.h"
#include "EinsumsInCpp/Print.hpp"
#include "EinsumsInCpp/State.hpp"
#include "EinsumsInCpp/Timer.hpp"

#include <catch2/catch.hpp>
#include <h5cpp/io>

auto main(int argc, char *argv[]) -> int {
    EinsumsInCpp::Timer::initialize();

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);

    // Create a file to hold the data from the DiskTensor tests.
    EinsumsInCpp::State::data = h5::create("Data.h5", H5F_ACC_TRUNC);

    EinsumsInCpp::println("Running on {} thread(s)", omp_get_max_threads());

    int result = Catch::Session().run(argc, argv);

    // EinsumsInCpp::Timer::report();
    EinsumsInCpp::Timer::finalize();
    return result;
}