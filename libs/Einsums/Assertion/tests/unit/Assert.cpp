//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Assert.hpp>

#include <source_location>
#include <string>
#include "Einsums/Debugging/Backtrace.hpp"

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

    result << "\n" << util::backtrace() << "\n";

    result_string = result.str();
}

TEST_CASE("assert") {
    using namespace einsums;

    einsums::detail::set_assertion_handler(test_assertion_handler);

    result_string = "";

    ((bool)(true) ? void() : ::einsums ::detail ::handle_assert(std ::source_location ::current(), "true", fmt ::format("")));

    REQUIRE(result_string == "");

    einsums::detail::set_assertion_handler(einsums::detail::default_assertion_handler);

}