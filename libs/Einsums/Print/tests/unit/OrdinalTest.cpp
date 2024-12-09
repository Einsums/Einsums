#include <Einsums/Testing.hpp>
#include <Einsums/Print.hpp>

TEST_CASE("Formatting ordinals", "[print]") {
    using namespace einsums;

    std::string formatted = fmt::format("{}", print::ordinal{1});

    REQUIRE(formatted == "1st");
}