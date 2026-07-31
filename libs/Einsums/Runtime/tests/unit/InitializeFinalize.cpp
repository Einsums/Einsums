//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Runtime.hpp>

#include <string>

#include <Einsums/Testing.hpp>

TEST_CASE("Initialize-Finalize", "[runtime]") {

    using namespace einsums;

    REQUIRE_NOTHROW(initialize(std::vector<std::string>{"einsums"}));

    SECTION("Normal finalize") {
        REQUIRE_NOTHROW(finalize());
    }

    SECTION("Double initialize/finalize") {
        REQUIRE_NOTHROW(initialize(std::vector<std::string>{"einsums"}));
        REQUIRE_NOTHROW(finalize());
        REQUIRE_NOTHROW(finalize());
        REQUIRE_NOTHROW(initialize(std::vector<std::string>{"einsums"}));
        REQUIRE_NOTHROW(finalize());
    }
}