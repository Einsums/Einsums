#define CATCH_CONFIG_RUNNER
#include "EinsumsInCpp/State.hpp"
#include "EinsumsInCpp/Timer.hpp"

#include <catch2/catch.hpp>
#include <h5cpp/io>

auto main(int argc, char *argv[]) -> int {
    EinsumsInCpp::Timer::initialize();

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);

    // Create the integral file in the temporary directory
    // This does not delete the file. Since the integral file can grow large
    // in size this should be considered.
    EinsumsInCpp::State::data = h5::create("Data.h5", H5F_ACC_TRUNC);

    int result = Catch::Session().run(argc, argv);

    EinsumsInCpp::Timer::finalize();
    return result;
}