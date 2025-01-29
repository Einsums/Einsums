#include <Einsums/Runtime/InitializeFinalize.hpp>

#include <string>

#include <Einsums/Testing.hpp>

TEST_CASE("Initialize-Finalize", "[runtime]") {

    using namespace einsums;

    REQUIRE(initialize_testing() == 0);

    SECTION("Normal finalize") {
        REQUIRE_NOTHROW(finalize());
    }

    SECTION("Stringify finalize") {
        std::stringstream stream;

        REQUIRE_NOTHROW(finalize(stream));

        std::string output = stream.str();

        REQUIRE(output.size() != 0);
    }
}