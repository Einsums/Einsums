#include <Einsums/Errors/Error.hpp>

#include "Einsums/Errors/ThrowException.hpp"

#include <Einsums/Testing.hpp>

static void thrower() {
    EINSUMS_THROW_EXCEPTION(einsums::error::no_success, "Test error.");
}

TEST_CASE("Test catching", "[error]") {
    REQUIRE_THROWS_AS(thrower(), einsums::error::no_success);
}