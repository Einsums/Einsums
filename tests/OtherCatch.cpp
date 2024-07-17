#include "einsums/Utilities.hpp"

#include <catch2/catch_all.hpp>

TEMPLATE_TEST_CASE("Strict floating-point matchers", "[matchers]", float, double) {
    using namespace einsums;

    auto matcher = einsums::WithinStrict(TestType{0.0}, TestType{1000.0});

    CHECK_FALSE(matcher.match(TestType{1.0}));
    CHECK(matcher.match(0.0));

    double a, b, c;

    CHECK(std::sscanf(matcher.describe().c_str(), "%lf%lf%lf", &a, &b, &c) == 3);
}