#include <Einsums/Utilities/Random.hpp>
#include <Einsums/Print.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Test random number generation", "[randomness]", float, double) {
    using namespace einsums;

    detail::circle_distribution<TestType> dist(0, 1);

    TestType expected_mean = 0.0, expected_median = 0, expected_variance = 1.0 / 3.0, expected_kurtosis = -6.0 / 5.0;

    size_t constexpr points = 1000000;

    std::vector<TestType> data(points);

    REQUIRE(dist(einsums::random_engine) != dist(einsums::random_engine));

    for (size_t i = 0; i < points; i++) {
        data[i] = dist(einsums::random_engine);
    }

    for(size_t i = 0; i < 10; i++) {
        println("{}", data[i]);
    }

    // Check to make sure everything is within range.
    for (size_t i = 0; i < points; i++) {
        REQUIRE(data[i] < TestType{1.0});
        REQUIRE(data[i] > TestType{-1.0});
    }

    // Check to make sure the mean is about right.
    TestType mean = 0.0;

#pragma omp parallel for reduction(+ : mean)
    for (size_t i = 0; i < points; i++) {
        mean += data[i];
    }

    mean /= points;

    REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(expected_mean, TestType{1e-3}));

    // Check to make sure the variance is about right.
    TestType variance = 0.0;

#pragma omp parallel for reduction(+ : variance)
    for (size_t i = 0; i < points; i++) {
        variance += (data[i] - mean) * (data[i] - mean);
    }

    TestType m2 = variance;

    variance /= points - 1;

    REQUIRE_THAT(variance, Catch::Matchers::WithinRel(expected_variance, TestType{1e-2}));

    // Check that the kurtosis is about right.
    TestType kurtosis = 0.0;

#pragma omp parallel for reduction(+ : kurtosis)
    for (size_t i = 0; i < points; i++) {
        kurtosis += (data[i] - mean) * (data[i] - mean) * (data[i] - mean) * (data[i] - mean);
    }

    // Excess kurtosis is a complicated formula.
    kurtosis = (kurtosis / (m2 * m2)) * (points + 1) * points * (points - 1) / (TestType)((points - 2) * (points - 3)) -
               3.0 * (points - 1) * (points - 1) / (TestType)((points - 2) * (points - 3));

    CHECK_THAT(kurtosis, Catch::Matchers::WithinRel(expected_kurtosis, TestType{1e-2}));

    // Check to see that the median is about right.
    std::sort(data.begin(), data.end());

    TestType median = (data[points / 2] + data[points / 2 + 1]) / 2;

    REQUIRE_THAT(median, Catch::Matchers::WithinAbs(expected_median, TestType{1e-3}));
}