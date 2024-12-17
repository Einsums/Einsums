//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Profile/Timer.hpp>
#include <Einsums/Runtime/InitializeFinalize.hpp>
#include <Einsums/Utilities/Random.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUStreams/GPUStreams.hpp>
#endif

#include <h5cpp/all>
#include <string>

namespace einsums {

int initialize() {
    error::initialize();

#ifdef EINSUMS_COMPUTE_CODE
    einsums::gpu::initialize();
#endif

    profile::initialize();
    blas::initialize();

    // Disable HDF5 diagnostic reporting
    H5Eset_auto(0, nullptr, nullptr);

    einsums::random_engine = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());

    return 0;
}

namespace {
void finalize_pre(void) {
    blas::finalize();

#ifdef EINSUMS_COMPUTE_CODE
    einsums::gpu::finalize();
#endif
}

void finalize_post(void) {
    profile::finalize();
}
} // namespace

void finalize(char const *output_file) {
    auto fp = std::fopen(output_file, "w");

    finalize(fp);

    std::fclose(fp);
}

void finalize(bool timerReport) {
    if (timerReport) {
        finalize(std::cout);
    }
}

void finalize(std::string const &output_file) {
    auto fp = std::ofstream(output_file);

    finalize(fp);

    fp.close();
}

void finalize(FILE *file_pointer) {
    finalize_pre();

    profile::report(file_pointer);

    finalize_post();
}

void finalize(std::ostream &output_stream) {
    finalize_pre();

    profile::report(output_stream);

    finalize_post();
}

} // namespace einsums