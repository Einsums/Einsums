#include "einsums/polynomial/Laguerre.hpp"
#include "einsums/polynomial/Utilities.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("gauss_laguerre3") {
    using namespace einsums;

    auto [x, w] = polynomial::laguerre::gauss_laguerre(3);

    CHECK_THAT(x.vector_data(), Catch::Matchers::Approx(
                                    std::vector<double, einsums::AlignedAllocator<double, 64>>{0.4157745568, 2.2942803603, 6.2899450829}));
    CHECK_THAT(w.vector_data(), Catch::Matchers::Approx(
                                    std::vector<double, einsums::AlignedAllocator<double, 64>>{0.7110930099, 0.2785177336, 0.0103892565}));
}

TEST_CASE("gauss_laguerre10") {
    using namespace einsums;

    auto [x, w] = polynomial::laguerre::gauss_laguerre(10);

    CHECK_THAT(x.vector_data(), Catch::Matchers::Approx(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                    0.13779347, 0.72945455, 1.8083429, 3.4014337, 5.55249614, 8.33015275, 11.84378584, 16.27925783,
                                    21.99658581, 29.92069701}));
    CHECK_THAT(w.vector_data(), Catch::Matchers::Approx(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                    3.08441116e-01, 4.01119929e-01, 2.18068288e-01, 6.20874561e-02, 9.50151698e-03, 7.53008389e-04,
                                    2.82592335e-05, 4.24931398e-07, 1.83956482e-09, 9.91182722e-13}));
}
