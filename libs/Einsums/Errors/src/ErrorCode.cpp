//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ErrorCode.hpp>
#include <Einsums/Errors/Exception.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

namespace einsums {
namespace detail {

struct EinsumsCategory : std::error_category {
    [[nodiscard]] auto name() const noexcept -> char const * override { return "einsums"; }

    [[nodiscard]] auto message(int value) const -> std::string override {
        if (value >= static_cast<int>(einsums::Error::success) && value < static_cast<int>(einsums::Error::last_error)) {
            return std::string("einsums(") + error_names[value] + ")";
        }
        if (error_code_has_system_error(value)) {
            return {"einsums(system_error)"};
        }
        return "einsums(unknown_error)";
    }
};

struct einsums_category_rethrow : std::error_category {
    [[nodiscard]] auto name() const noexcept -> char const * override { return ""; }

    [[nodiscard]] auto message(int) const noexcept -> std::string override { return ""; }
};

} // namespace detail

auto get_einsums_category() -> std::error_category const & {
    static detail::EinsumsCategory einsums_category;
    return einsums_category;
}

auto get_einsums_rethrow_category() -> std::error_category const & {
    static detail::einsums_category_rethrow einsums_category_rethrow;
    return einsums_category_rethrow;
}

namespace detail {

auto get_einsums_category(ThrowMode mode) -> std::error_category const & {
    switch (mode) {
    case ThrowMode::rethrow:
        return get_einsums_rethrow_category();

    case ThrowMode::plain:
    default:
        break;
    }
    return einsums::get_einsums_category();
}

} // namespace detail

ErrorCode::ErrorCode(Error e, ThrowMode mode) : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::Error::success && e != einsums::Error::no_success) {
        _exception = detail::get_exception(e, "", mode);
    }
}

ErrorCode::ErrorCode(Error e, std::source_location const &location, ThrowMode mode)
    : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::Error::success && e != einsums::Error::no_success) {
        _exception = detail::get_exception(e, "", mode, location);
    }
}

ErrorCode::ErrorCode(Error e, char const *msg, ThrowMode mode) : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::Error::success && e != einsums::Error::no_success) {
        _exception = detail::get_exception(e, msg, mode);
    }
}

ErrorCode::ErrorCode(Error e, char const *msg, std::source_location const &location, ThrowMode mode)
    : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::Error::success && e != einsums::Error::no_success) {
        _exception = detail::get_exception(e, msg, mode, location);
    }
}

ErrorCode::ErrorCode(Error e, std::string const &msg, ThrowMode mode) : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::Error::success && e != einsums::Error::no_success) {
        _exception = detail::get_exception(e, msg, mode);
    }
}

ErrorCode::ErrorCode(Error e, std::string const &msg, std::source_location const &location, ThrowMode mode)
    : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::Error::success && e != einsums::Error::no_success) {
        _exception = detail::get_exception(e, msg, mode, location);
    }
}

ErrorCode::ErrorCode(int err, Exception const &e) {
    assign(err, get_einsums_category());
    _exception = std::make_exception_ptr(e);
}

ErrorCode::ErrorCode(std::exception_ptr const &e)
    : std::error_code(detail::make_system_error_code(get_error(e), ThrowMode::rethrow)), _exception(e) {
}

auto ErrorCode::get_message() const -> std::string {
    if (_exception) {
        try {
            std::rethrow_exception(_exception);
        } catch (std::exception const &be) {
            return be.what();
        }
    }
    return get_error_what(*this);
}

ErrorCode::ErrorCode(ErrorCode const &rhs)
    : std::error_code(static_cast<einsums::Error>(rhs.value()) == Error::success ? make_success_code(ThrowMode::plain) : rhs),
      _exception(rhs._exception) {
}

auto ErrorCode::operator=(ErrorCode const &rhs) -> ErrorCode & {
    if (this != &rhs) {
        if (static_cast<Error>(rhs.value()) == Error::success) {
            // if the rhs is a success code, we maintain our throw mode
            this->std::error_code::operator=(make_success_code(ThrowMode::plain));
        } else {
            this->std::error_code::operator=(rhs);
        }
        _exception = rhs._exception;
    }
    return *this;
}

} // namespace einsums