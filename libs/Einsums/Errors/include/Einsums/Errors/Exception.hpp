//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ErrorCode.hpp>
#include <Einsums/Errors/ExceptionForward.hpp>
#include <Einsums/Errors/ExceptionInfo.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <system_error>
#include <utility>

namespace einsums {

struct EINSUMS_EXPORT Exception : std::system_error {
    explicit Exception(Error e = Error::success);

    explicit Exception(std::system_error const &e);

    explicit Exception(std::error_code const &e);

    Exception(Error e, char const *msg, ThrowMode mode = ThrowMode::plain);

    Exception(Error e, std::string const &msg, ThrowMode mode = ThrowMode::plain);

    ~Exception() noexcept override;

    [[nodiscard]] auto get_error() const noexcept -> Error;

    [[nodiscard]] auto get_error_code(ThrowMode mode = ThrowMode::plain) const noexcept -> ErrorCode;
};

namespace detail {

using custom_exception_info_handler_type = std::function<einsums::ExceptionInfo(std::source_location const &, std::string const &)>;

EINSUMS_EXPORT void set_custom_exception_info_handler(custom_exception_info_handler_type f);

using pre_exception_handler_type = std::function<void()>;

EINSUMS_EXPORT void set_pre_exception_handler(pre_exception_handler_type f);

} // namespace detail

namespace detail {

EINSUMS_DEFINE_ERROR_INFO(ThrowFunction, std::string);
EINSUMS_DEFINE_ERROR_INFO(ThrowFile, std::string);
EINSUMS_DEFINE_ERROR_INFO(ThrowLine, long);

struct EINSUMS_EXPORT std_exception : std::exception {
    explicit std_exception(std::string w) : _what(std::move(w)) {}

    ~std_exception() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_alloc : std::bad_alloc {
    explicit bad_alloc(std::string w) : _what(std::move(w)) {}

    ~bad_alloc() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_exception : std::bad_exception {

  public:
    explicit bad_exception(std::string w) : _what(std::move(w)) {}

    ~bad_exception() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_cast : std::bad_cast {

  public:
    explicit bad_cast(std::string w) : _what(std::move(w)) {}

    ~bad_cast() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_typeid : std::bad_typeid {
    explicit bad_typeid(std::string w) : _what(std::move(w)) {}

    ~bad_typeid() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> char const * override { return _what.c_str(); }

  private:
    std::string _what;
};

template <typename Exception>
EINSUMS_EXPORT auto get_exception(einsums::Exception const &e, std::string const &func, std::string const &file, long line,
                                  std::string const &auxinfo) -> std::exception_ptr;

template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e) -> std::exception_ptr;

} // namespace detail

EINSUMS_EXPORT auto get_error_what(ExceptionInfo const &xi) -> std::string;

template <typename E>
auto get_error_what(E const &e) -> std::string {
    return invoke_with_exception_info(e, [](ExceptionInfo const *xi) { return xi ? get_error_what(*xi) : std::string("<unknown>"); });
}

inline auto get_error_what(ErrorCode const &e) -> std::string {
    return get_error_what<ErrorCode>(e);
}

inline auto get_error_what(std::exception const &e) -> std::string {
    return e.what();
}

EINSUMS_EXPORT auto get_error(Exception const &e) -> Error;

EINSUMS_EXPORT auto get_error(ErrorCode const &e) -> Error;

EINSUMS_EXPORT auto get_error(std::exception_ptr const &e) -> Error;

EINSUMS_EXPORT auto get_error_function_name(einsums::ExceptionInfo const &xi) -> std::string;

template <typename E>
auto get_error_function_name(E const &e) -> std::string {
    return invoke_with_exception_info(e,
                                      [](ExceptionInfo const *xi) { return xi ? get_error_function_name(*xi) : std::string("<unknown>"); });
}

EINSUMS_EXPORT auto get_error_file_name(einsums::ExceptionInfo const &xi) -> std::string;

template <typename E>
auto get_error_file_name(E const &e) -> std::string {
    return invoke_with_exception_info(e, [](ExceptionInfo const *xi) { return xi ? get_error_file_name(*xi) : std::string("<unknown>"); });
}

EINSUMS_EXPORT auto get_error_line_number(einsums::ExceptionInfo const &xi) -> long;

template <typename E>
auto get_error_line_number(E const &e) -> long {
    return invoke_with_exception_info(e, [](ExceptionInfo const *xi) { return xi ? get_error_line_number(*xi) : -1; });
}

} // namespace einsums

#include <Einsums/Errors/ThrowException.hpp>
