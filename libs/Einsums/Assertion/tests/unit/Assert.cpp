//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/Backtrace.hpp>

#include <source_location>
#include <string>

#include <Einsums/Testing.hpp>

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

    einsums::detail::set_assertion_handler(test_assertion_handler);

    result_string = "";

    SECTION("True") {
        EINSUMS_ASSERT(true);

        REQUIRE(result_string == "");
    }

    SECTION("False") {
        EINSUMS_ASSERT(false);

#ifdef EINSUMS_DEBUG
        REQUIRE(result_string != "");
#else
        REQUIRE(result_string == "");
#endif
    }

    einsums::detail::set_assertion_handler(einsums::detail::default_assertion_handler);
}