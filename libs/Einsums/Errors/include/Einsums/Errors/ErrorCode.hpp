//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ExceptionForward.hpp>

#include <exception>
#include <source_location>
#include <string>
#include <system_error>

namespace einsums {

namespace detail {

EINSUMS_EXPORT auto access_exception(ErrorCode const &) -> std::exception_ptr;

} // namespace detail

EINSUMS_EXPORT auto get_einsums_category() -> std::error_category const &;

EINSUMS_EXPORT auto get_einsums_rethrow_category() -> std::error_category const &;

namespace detail {

EINSUMS_EXPORT auto get_einsums_category(ThrowMode mode) -> std::error_category const &;

inline auto make_system_error_code(Error e, ThrowMode mode = ThrowMode::plain) -> std::error_code {
    return {static_cast<int>(e), get_einsums_category(mode)};
}

inline auto make_error_condition(Error e, ThrowMode mode) -> std::error_condition {
    return {static_cast<int>(e), get_einsums_category(mode)};
}

} // namespace detail

struct ErrorCode : std::error_code {
    explicit ErrorCode(ThrowMode const mode = ThrowMode::plain) : std::error_code(detail::make_system_error_code(Error::success, mode)) {}

    EINSUMS_EXPORT explicit ErrorCode(Error e, ThrowMode mode = ThrowMode::plain);

    EINSUMS_EXPORT ErrorCode(Error e, std::source_location const &location = std::source_location::current(),
                             ThrowMode mode = ThrowMode::plain);

    EINSUMS_EXPORT ErrorCode(Error e, char const *msg, ThrowMode mode = ThrowMode::plain);

    EINSUMS_EXPORT ErrorCode(Error e, char const *msg, std::source_location const &location = std::source_location::current(),
                             ThrowMode mode = ThrowMode::plain);

    EINSUMS_EXPORT ErrorCode(Error e, std::string const &msg, ThrowMode mode = ThrowMode::plain);

    EINSUMS_EXPORT ErrorCode(Error e, std::string const &msg, std::source_location const &location = std::source_location::current(),
                             ThrowMode mode = ThrowMode::plain);

    EINSUMS_EXPORT auto get_message() const -> std::string;

    void clear() {
        assign(static_cast<int>(Error::success), get_einsums_category());
        _exception = std::exception_ptr();
    }

    EINSUMS_EXPORT ErrorCode(ErrorCode const &rhs);

    EINSUMS_EXPORT auto operator=(ErrorCode const &rhs) -> ErrorCode &;

  private:
    friend auto detail::access_exception(ErrorCode const &) -> std::exception_ptr;
    friend struct Exception;
    friend auto make_error_code(std::exception_ptr const &) -> ErrorCode;

    EINSUMS_EXPORT ErrorCode(int err, Exception const &e);
    EINSUMS_EXPORT explicit ErrorCode(std::exception_ptr const &e);

    std::exception_ptr _exception;
};

inline auto make_error_code(Error e, ThrowMode mode = ThrowMode::plain) -> ErrorCode {
    return ErrorCode(e, mode);
}

inline auto make_error_code(Error e, std::source_location const &location, ThrowMode mode = ThrowMode::plain) -> ErrorCode {
    return {e, location, mode};
}

inline auto make_error_code(Error e, char const *msg, ThrowMode mode = ThrowMode::plain) -> ErrorCode {
    return {e, msg, mode};
}

inline auto make_error_code(Error e, char const *msg, std::source_location const &location, ThrowMode mode = ThrowMode::plain)
    -> ErrorCode {
    return {e, msg, location, mode};
}

inline auto make_error_code(Error e, std::string const &msg, ThrowMode mode = ThrowMode::plain) -> ErrorCode {
    return {e, msg, mode};
}

inline auto make_error_code(Error e, std::string const &msg, std::source_location const &location, ThrowMode mode = ThrowMode::plain)
    -> ErrorCode {
    return {e, msg, location, mode};
}

inline auto make_error_code(std::exception_ptr const &e) -> ErrorCode {
    return ErrorCode(e);
}

inline auto make_success_code(ThrowMode mode = ThrowMode::plain) -> ErrorCode {
    return ErrorCode(mode);
}

} // namespace einsums