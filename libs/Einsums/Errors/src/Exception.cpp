//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ErrorCode.hpp>
#include <Einsums/Errors/Exception.hpp>
#include <Einsums/Errors/ExceptionInfo.hpp>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace einsums {

Exception::Exception(Error e) : std::system_error(make_error_code(e, ThrowMode::plain)) {
    EINSUMS_ASSERT((e >= einsums::Error::success && e < einsums::Error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
}

Exception::Exception(std::system_error const &e) : std::system_error(e) {
}

Exception::Exception(std::error_code const &e) : std::system_error(e) {
}

Exception::Exception(Error e, char const *msg, ThrowMode mode) : std::system_error(detail::make_system_error_code(e, mode), msg) {
    EINSUMS_ASSERT((e >= einsums::Error::success && e < einsums::Error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
}

Exception::Exception(Error e, std::string const &msg, ThrowMode mode) : std::system_error(detail::make_system_error_code(e, mode), msg) {
    EINSUMS_ASSERT((e >= einsums::Error::success && e < einsums::Error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
}

Exception::~Exception() noexcept = default;

auto Exception::get_error() const noexcept -> Error {
    return static_cast<Error>(code().value());
}

auto Exception::get_error_code(ThrowMode mode) const noexcept -> ErrorCode {
    (void)mode;
    return {code().value(), *this};
}

namespace detail {

namespace {
custom_exception_info_handler_type custom_exception_info_handler;
pre_exception_handler_type         pre_exception_handler;
} // namespace

void set_custom_exception_info_handler(custom_exception_info_handler_type f) {
    custom_exception_info_handler = std::move(f);
}

void set_pre_exception_handler(pre_exception_handler_type f) {
    pre_exception_handler = std::move(f);
}

template <typename Exception>
EINSUMS_EXPORT auto construct_custom_exception(Exception const &e, std::source_location const &location, std::string const &auxinfo)
    -> std::exception_ptr {
    // create a std::exception_ptr object encapsulating the Exception to
    // be thrown and annotate it with information provided by the hook
    try {
        throw_with_info(e, custom_exception_info_handler(location, auxinfo));
    } catch (...) {
        return std::current_exception();
    }

    return {};
}

auto access_exception(ErrorCode const &e) -> std::exception_ptr {
    return e._exception;
}

template <typename Exception>
EINSUMS_EXPORT auto get_exception(Exception const &e, std::source_location const &location, std::string const &auxinfo)
    -> std::exception_ptr {
    return construct_custom_exception(e, location, auxinfo);
}

template <typename Exception>
EINSUMS_EXPORT void throw_exception(Exception const &e, std::source_location const &location) {
    if (pre_exception_handler) {
        pre_exception_handler();
    }

    std::rethrow_exception(get_exception(e, location));
}

template EINSUMS_EXPORT auto get_exception(Exception const &, std::source_location const &, std::string const &) -> std::exception_ptr;

template EINSUMS_EXPORT auto get_exception(std::system_error const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;

template EINSUMS_EXPORT auto get_exception(std::exception const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std_exception const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_exception const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(bad_exception const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_typeid const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(bad_typeid const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_cast const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(bad_cast const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_alloc const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(bad_alloc const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::logic_error const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::runtime_error const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::out_of_range const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::invalid_argument const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;

template EINSUMS_EXPORT void throw_exception(Exception const &, std::source_location const &);

template EINSUMS_EXPORT void throw_exception(std::system_error const &, std::source_location const &);

template EINSUMS_EXPORT void throw_exception(std::exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std_exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(bad_exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_typeid const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(bad_typeid const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_cast const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(bad_cast const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_alloc const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(bad_alloc const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::logic_error const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::runtime_error const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::out_of_range const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::invalid_argument const &, std::source_location const &);

} // namespace detail

auto get_error_what(einsums::ExceptionInfo const &xi) -> std::string {
    auto const *se = dynamic_cast<std::exception const *>(&xi);
    return se ? se->what() : "<unknown>";
}

auto get_error(Exception const &e) -> Error {
    return static_cast<Error>(e.get_error());
}

auto get_error(std::exception_ptr const &e) -> Error {
    try {
        std::rethrow_exception(e);
    } catch (Exception const &he) {
        return he.get_error();
    } catch (std::system_error const &e) {
        int code = e.code().value();
        if (code < static_cast<int>(einsums::Error::success) || code >= static_cast<int>(einsums::Error::last_error)) {
            code |= static_cast<int>(einsums::Error::system_error_flag);
        }
        return static_cast<einsums::Error>(code);
    } catch (...) {
        return einsums::Error::unknown_error;
    }
}

auto get_error_function_name(einsums::ExceptionInfo const &xi) -> std::string {
    std::string const *function = xi.get<einsums::detail::ThrowFunction>();
    if (function) {
        return *function;
    }

    return {};
}

auto get_error_file_name(einsums::ExceptionInfo const &xi) -> std::string {
    std::string const *file = xi.get<einsums::detail::ThrowFile>();
    if (file) {
        return *file;
    }

    return "<unknown>";
}

auto get_error_line_number(einsums::ExceptionInfo const &xi) -> long {
    long const *line = xi.get<einsums::detail::ThrowLine>();
    if (line) {
        return *line;
    }
    return -1;
}

} // namespace einsums