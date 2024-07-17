//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#define CATCH_CONFIG_RUNNER
#include "einsums/_Compiler.hpp"

#include "einsums/Blas.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Timer.hpp"

#include <catch2/catch_all.hpp>
#include <h5cpp/io>

auto main(int argc, char *argv[]) -> int {
    Catch::StringMaker<float>::precision = 10;
    Catch::StringMaker<double>::precision = 17;
    einsums::initialize();

    // Create a file to hold the data from the DiskTensor tests.
    EINSUMS_DISABLE_WARNING_PUSH
    EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
    einsums::state::data() = h5::create(std::string(std::tmpnam(nullptr)), H5F_ACC_TRUNC);
    EINSUMS_DISABLE_WARNING_POP

    // println("Running on {} thread(s)", omp_get_max_threads());

    int result = Catch::Session().run(argc, argv);

    // Ensure file is closed before finalize is called. If einsums is running parallel MPI will have been
    // finalized BEFORE the HDF5 file has been closed and MPI will report that an MPI function was called
    // after having been finalized.
    H5Fclose(einsums::state::data());

    // Passing false means "do not print timer report", whereas passing true will print the
    // timer report.
    // Print the timer report to a file anyways.
    einsums::timer::report("./timings.txt");
    einsums::finalize(false);

    return result;
}
