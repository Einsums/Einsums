//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/Exception.hpp>

#include <exception>
#include <filesystem>
#include <string>
#include <system_error>

namespace einsums {
ErrorCode throws; // "throw on error" special error_code;
}

namespace einsums::detail {

[[noreturn]] void throw_exception(Error errcode, std::string const &msg, std::source_location const &location) {
    detail::throw_exception(Exception(errcode, msg, ThrowMode::plain), location);
}

[[noreturn]] void rethrow_exception(Exception const &e, std::source_location const &location) {
    detail::throw_exception(Exception(e.get_error(), e.what(), ThrowMode::plain), location);
}

auto get_exception(Error errcode, std::string const &msg, ThrowMode mode, std::source_location const &location, std::string const &auxinfo)
    -> std::exception_ptr {
    return get_exception(Exception(errcode, msg, mode), location, auxinfo);
}

auto get_exception(std::error_code const &ec, std::string const & /* msg */, ThrowMode /* mode */, std::source_location const &location,
                   std::string const &auxinfo) -> std::exception_ptr {
    return get_exception(Exception(ec), location, auxinfo);
}

void throws_if(ErrorCode &ec, Error errcode, std::string const &msg, std::source_location const &location) {
    if (&ec == &throws) {
        detail::throw_exception(errcode, msg, location);
    } else {
        ec = make_error_code(errcode, msg, location, ThrowMode::plain);
    }
}

void rethrows_if(ErrorCode &ec, Exception const &e, std::source_location const &location) {
    if (&ec == &throws) {
        detail::rethrow_exception(e, location);
    } else {
        ec = make_error_code(e.get_error(), e.what(), location, ThrowMode::rethrow);
    }
}

} // namespace einsums::detail