#include <Einsums/Testing.hpp>
#include <Einsums/Print.hpp>

TEST_CASE("Formatting ordinals", "[print]") {
    using namespace einsums;

    print::ordinal<int> first = 1;

    std::string formatted = fmt::format("{}", first);

    REQUIRE(formatted == "1st");
}