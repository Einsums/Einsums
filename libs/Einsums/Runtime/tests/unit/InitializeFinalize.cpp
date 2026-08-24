//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/Runtime.hpp>

#include <string>

#include <Einsums/Testing.hpp>

TEST_CASE("Initialize-Finalize", "[runtime]") {

    EINSUMS_LOG_DEBUG("Starting initialize test.");

    using namespace einsums;

    EINSUMS_LOG_DEBUG("Creating argument vector.");
    auto args = std::vector<std::string>{"einsums"};
    
    EINSUMS_LOG_DEBUG("Reinitializing the library.");

    REQUIRE_NOTHROW(initialize(args));

    EINSUMS_LOG_DEBUG("Initialized again.");

    SECTION("Normal finalize") {

        EINSUMS_LOG_DEBUG("Finalizing Einsums.");
        REQUIRE_NOTHROW(finalize());

        EINSUMS_LOG_DEBUG("Finished with section.");
    }

    SECTION("Double initialize/finalize") {
        EINSUMS_LOG_DEBUG("Initializing.");
        REQUIRE_NOTHROW(initialize(args));
        EINSUMS_LOG_DEBUG("Finalizing.");
        REQUIRE_NOTHROW(finalize());
        EINSUMS_LOG_DEBUG("Finalizing.");
        REQUIRE_NOTHROW(finalize());
        EINSUMS_LOG_DEBUG("Finalizing.");
        EINSUMS_LOG_DEBUG("Initializing.");
        REQUIRE_NOTHROW(initialize(args));
        EINSUMS_LOG_DEBUG("Finalizing.");
        REQUIRE_NOTHROW(finalize());
        EINSUMS_LOG_DEBUG("Finished with section.")
    }
}