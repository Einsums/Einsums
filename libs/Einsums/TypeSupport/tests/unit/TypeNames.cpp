#include "Einsums/TypeSupport/TypeName.hpp"
#include "catch2/catch_test_macros.hpp"

#include <Einsums/Testing.hpp>

TEST_CASE("Type Names") {
    INFO(einsums::detail::get_type_name_string_view<int>());

    REQUIRE(einsums::detail::get_type_name_string_view<int>() == "int");
}