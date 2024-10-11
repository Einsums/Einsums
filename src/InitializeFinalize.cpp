//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/_Common.hpp"

#include "einsums/Blas.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Timer.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

#include <h5cpp/all>

namespace einsums {

auto initialize() -> int {
#ifdef __HIP__
    einsums::gpu::initialize();
#endif

    timer::initialize();
    blas::initialize();

    // Disable nested omp regions
    // omp_set_max_active_levels(1);

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);

    return 0;
}

static void finalize_pre(void) {
        blas::finalize();

#ifdef __HIP__
    einsums::gpu::finalize();

#endif
}

static void finalize_post(void) {
    timer::finalize();
}

void finalize(const char *output_file) {
    finalize_pre();


    timer::report(output_file);

    finalize_post();
}

void finalize(bool timerReport) {

    finalize_pre();

    if (timerReport)
        timer::report();

    finalize_post();
}

void finalize(const std::string &output_file) {
    finalize_pre();


    timer::report(output_file);

    finalize_post();
}

void finalize(FILE *file_pointer) {
    finalize_pre();


    timer::report(file_pointer);

    finalize_post();
}

void finalize(std::ostream &output_stream) {
    finalize_pre();


    timer::report(output_stream);

    finalize_post();
}

} // namespace einsums
