//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <fmt/format.h>

#include <string>
#include <system_error>

namespace einsums {

enum class Error : std::uint16_t {
    /// The operation was successful
    success = 0,
    /// The operation failed, but not in an expected way
    no_success,
    unknown_error,
    bad_parameter,
    kernel_error,
    out_of_memory,
    invalid_status,
    tensors_incompatible,
    disk_error,
    assertion_failure,
    unhandled_exception,
    lock_error,

    last_error,

    system_error_flag = 0x4000L,
    error_upper_bound = 0x7fffL
};

namespace detail {

char const *const error_names[] = {"success",      "no_success",        "unknown_error",       "bad_parameter",
                                   "kernel_error", "out_of_memory",     "invalid_status",      "tensors_incompatible",
                                   "disk_error",   "assertion_failure", "unhandled_exception", "lock_error"};

inline auto error_code_has_system_error(int const e) -> bool {
    return e & static_cast<int>(Error::system_error_flag);
}

} // namespace detail

} // namespace einsums

template <>
struct fmt::formatter<einsums::Error> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(einsums::Error e, FormatContext &ctx) const {
        int e_int = static_cast<int>(e);
        if (e_int >= static_cast<int>(einsums::Error::success) && e_int < static_cast<int>(einsums::Error::last_error)) {
            return fmt::formatter<std::string>::format(einsums::detail::error_names[e_int], ctx);
        } else {
            return fmt::formatter<std::string>::format(fmt::format("invalid einsums::error ({})", e_int), ctx);
        }
    }
};

namespace std {

template <>
struct is_error_code_enum<einsums::Error> {
    static constexpr bool value = true;
};

template <>
struct is_error_condition_enum<einsums::Error> {
    static constexpr bool value = true;
};

} // namespace std