#include "einsums/parallel/MPI.hpp"

#include <catch2/catch_all.hpp>

using namespace einsums::mpi;

#if 0
TEST_CASE("mpi-init") {
    {
        ErrorOr<bool, Error> error = initialized();
        REQUIRE_FALSE(error.is_error());
        REQUIRE(error.value() == false);
    }
    {
        ErrorOr<void, Error> error = initialize(0, nullptr);
        REQUIRE_FALSE(error.is_error());
    }
    {
        ErrorOr<bool, Error> error = initialized();
        REQUIRE_FALSE(error.is_error());
        REQUIRE(error.value() == true);
    }
    {
        ErrorOr<bool, Error> error = finalized();
        REQUIRE_FALSE(error.is_error());
        REQUIRE(error.value() == false);
    }
    {
        ErrorOr<void, Error> error = finalize();
        REQUIRE_FALSE(error.is_error());
    }
    {
        ErrorOr<bool, Error> error = finalized();
        REQUIRE_FALSE(error.is_error());
        REQUIRE(error.value() == true);
    }
}

TEST_CASE("mpi-noinit") {
    {
        ErrorOr<bool, Error> error = initialized();
        REQUIRE_FALSE(error.is_error());
        REQUIRE(error.value() == false);
    }
}
#endif

TEST_CASE("mpi-thread") {
    // MPI should already be initialized
    {
        ErrorOr<bool> error = initialized();
        REQUIRE_FALSE(error.is_error());
        REQUIRE(error.value() == true);
    }

    {
        ErrorOr<ThreadLevel> result = query_thread();
        REQUIRE_FALSE(result.is_error());

        println("Thread level: {}", result.value());
    }
}
