#include "einsums/Utilities.hpp"

#include <catch2/catch_all.hpp>

TEMPLATE_TEST_CASE("Strict floating-point matchers", "[matchers]", float, double) {
    using namespace einsums;

    auto matcher = einsums::WithinStrict(TestType{0.0}, TestType{1000.0});

    CHECK_FALSE(matcher.match(TestType{1.0}));
    CHECK(matcher.match(0.0));
    CHECK_THAT(matcher.describe(), Catch::Matchers::Matches("is within a fraction of -?[0-9]+(\\.[0-9]*)?f? to -?[0-9]+(\\.[0-9]*)?f?"));
}