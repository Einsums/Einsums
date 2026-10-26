//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/Backtrace.hpp>
#include <Einsums/Logging.hpp>

#include <source_location>
#include <string>

#include <Einsums/Testing.hpp>

#ifdef EINSUMS_WITH_BACKTRACES
#    include <cpptrace/cpptrace.hpp>
#endif

static std::string result_string;

void test_assertion_handler(std::source_location const &loc, char const *expr, std::string const &msg) {
    using namespace einsums;
    std::ostringstream result;
    result << loc.function_name() << ":" << loc.line() << " : Assertion '" << expr << "' failed";
    if (!msg.empty()) {
        result << " (" << msg << ")\n";
    } else {
        result << "\n";
    }

#ifdef EINSUMS_HAVE_BACKTRACES
    std::string backtrace;

    result << "\n";

    util::print_backtrace(result);

    result << "\n";

#endif

    result_string = result.str();
}

TEST_CASE("assert") {
    using namespace einsums;

    EINSUMS_LOG_INFO("Setting assertion handler.");

    einsums::detail::set_assertion_handler(test_assertion_handler);

    EINSUMS_LOG_INFO("Setting the log string.");

    result_string = "";

    SECTION("True") {

        EINSUMS_LOG_INFO("Testing a true assertion.");
        EINSUMS_ASSERT(true);

        EINSUMS_LOG_INFO("Checking to see that that worked.");

        REQUIRE(result_string == "");
        EINSUMS_LOG_INFO("Done.");
    }

    SECTION("False") {
        EINSUMS_LOG_INFO("Testing a false assertion.");
        EINSUMS_ASSERT(false);

        EINSUMS_LOG_INFO("Checking to see if that worked.");

#ifdef EINSUMS_DEBUG
        REQUIRE(result_string != "");
#else
        REQUIRE(result_string == "");
#endif
        EINSUMS_LOG_INFO("Done.");
    }

    EINSUMS_LOG_INFO("Resetting the assertion handler.");
    einsums::detail::set_assertion_handler(einsums::detail::default_assertion_handler);
    EINSUMS_LOG_INFO("Calling destructors.");
}