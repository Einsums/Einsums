//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/parallel/MPI.hpp"

#include "einsums/Print.hpp"
#include "einsums/Section.hpp"
#include "einsums/_Common.hpp"

#include <mpi.h>
#include <stdexcept>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::mpi)

auto initialize(int *argc, char ***argv) -> ErrorOr<void> {
    // LabeledSection0() -> makes a call to VTune section routines which apparently detect that MPI is available.
    // LabeledSection0();
    int provided{0};
    EINSUMS_MPI_TEST(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided));
    if (provided != MPI_THREAD_MULTIPLE) {
        println_abort("MPI thread provided does not match requested: {} != {}", ThreadLevel(provided), ThreadLevel(MPI_THREAD_MULTIPLE));
    }
    return {};
}

auto finalize() -> ErrorOr<void> {
    LabeledSection0();
    EINSUMS_MPI_TEST(MPI_Finalize());
    return {};
}

auto initialized() -> ErrorOr<bool> {
    LabeledSection0();
    int result{0};
    EINSUMS_MPI_TEST(MPI_Initialized(&result));
    return static_cast<bool>(result);
}

auto finalized() -> ErrorOr<bool> {
    LabeledSection0();
    int result{0};
    EINSUMS_MPI_TEST(MPI_Finalized(&result));
    return static_cast<bool>(result);
}

auto query_thread() -> ErrorOr<ThreadLevel> {
    LabeledSection0();
    int result{0};
    EINSUMS_MPI_TEST(MPI_Query_thread(&result));
    return static_cast<ThreadLevel>(result);
}

auto size() -> ErrorOr<int> {
    LabeledSection0();
    int result{0};
    EINSUMS_MPI_TEST(MPI_Comm_size(MPI_COMM_WORLD, &result));
    return result;
}

auto rank() -> ErrorOr<int> {
    int result{0};
    EINSUMS_MPI_TEST(MPI_Comm_rank(MPI_COMM_WORLD, &result));
    return result;
}

auto Comm::group() const -> ErrorOr<Group> {
    Group v2;
    EINSUMS_MPI_TEST(MPI_Comm_group(_real_comm, &(v2._real_group)));
    return v2;
}

auto Comm::rank() const -> ErrorOr<int> {
    int v2{-1};
    EINSUMS_MPI_TEST(MPI_Comm_rank(_real_comm, &v2));
    return v2;
}

END_EINSUMS_NAMESPACE_CPP(einsums::mpi)