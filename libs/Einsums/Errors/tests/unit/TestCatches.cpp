#include <Einsums/Errors/Error.hpp>

#include "Einsums/Errors/ThrowException.hpp"

#include <Einsums/Testing.hpp>

static void thrower() {
    EINSUMS_THROW_EXCEPTION(einsums::dimension_error, "Test error.");
}

static void thrower2() {
    throw einsums::dimension_error("Test error2.");
}

TEST_CASE("Test catching", "[error]") {
    try {
        thrower2();
    } catch(const einsums::dimension_error &e) {
        REQUIRE(true);
    }

    try {
        thrower();
    } catch(einsums::dimension_error &e) {
        REQUIRE(true);
    }

    REQUIRE_THROWS_AS(thrower(), einsums::dimension_error);
}