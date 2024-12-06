#include <Einsums/Errors/Error.hpp>

#include "Einsums/Errors/ThrowException.hpp"

#include <Einsums/Testing.hpp>

static void thrower() {
    EINSUMS_THROW_EXCEPTION(einsums::dimension_error, "Test error.");
}

static void thrower2() {
    throw einsums::dimension_error("Test error2.");
}

static void thrower3() {
    throw std::logic_error("Logic Error test");
}

TEST_CASE("Test catching 1", "[error]") {
    try {
        thrower2();
    } catch(const einsums::dimension_error &e) {
        REQUIRE(true);
    }

    REQUIRE_THROWS_AS(thrower2(), einsums::dimension_error);
}

TEST_CASE("Test catching 2", "[error]") {
    try {
        thrower3();
    } catch(std::logic_error &e) {
        REQUIRE(true);
    }

    REQUIRE_THROWS_AS(thrower3(), std::logic_error);
}

TEST_CASE("Test catching 3", "[error]") {
    try {
        thrower();
    } catch(einsums::dimension_error &e) {
        REQUIRE(true);
    }

    REQUIRE_THROWS_AS(thrower(), einsums::dimension_error);
}