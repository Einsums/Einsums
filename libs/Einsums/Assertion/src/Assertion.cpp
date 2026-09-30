//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/Backtrace.hpp>
#include <Einsums/Version.hpp>

#include <iostream>
#include <mutex>

namespace {
einsums::detail::assertion_handler_type handler{einsums::detail::default_assertion_handler};

std::mutex handler_mutex;
} // namespace

namespace einsums::detail {

namespace {
auto get_handler() -> assertion_handler_type & {
    std::lock_guard lock(handler_mutex);

    if (handler == nullptr) {
        handler = default_assertion_handler;
    }

    return handler;
}
} // namespace

void default_assertion_handler(std::source_location const &loc, char const *expr, std::string const &msg) {
    std::cerr << complete_version() << std::endl << loc.function_name() << ":" << loc.line() << " : Assertion '" << expr << "' failed";
    if (!msg.empty()) {
        std::cerr << " (" << msg << ")" << std::endl;
    } else {
        std::cerr << std::endl;
    }

    std::cerr << std::endl;

    einsums::util::print_backtrace(std::cerr);

    std::cerr << std::endl;

    std::exit(EXIT_FAILURE);
}

void set_assertion_handler(assertion_handler_type handler_) {
    std::lock_guard lock(handler_mutex);

    if (handler_ == nullptr) {
        handler = default_assertion_handler;
    } else {
        handler = handler_;
    }
}

void handle_assert(std::source_location const &loc, char const *expr, std::string const &msg) noexcept {
    std::lock_guard lock(handler_mutex);

#ifdef EINSUMS_DEBUG
    std::cout << complete_version() << std::endl;
    std::cout << loc.function_name() << ": " << loc.line() << ": Assertion '" << expr << "' failed";
    if (!msg.empty()) {
        std::cout << " (" << msg << ')' << std::endl;
    } else {
        std::cout << std::endl;
    }
#endif

    if (handler == nullptr) {
        handler = default_assertion_handler;
    }

    handler(loc, expr, msg);
}

} // namespace einsums::detail
