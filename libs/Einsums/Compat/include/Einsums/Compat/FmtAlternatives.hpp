//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/*
 * Windows doesn't like certain library functions. We want to use library functions. This header provides wrappers for library functions
 * that don't throw errors on Windows.
 */

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/xchar.h>

#include <cstring>
#include <cwchar>
#include <string>

namespace einsums {
namespace detail {

// fmtlib doesn't have formatted_size for styled text. I'll just make my own.
template <typename... T>
FMT_NODISCARD FMT_INLINE auto formatted_size(fmt::text_style ts, fmt::format_string<T...> fmt, T &&...args) -> size_t {
    auto buf = fmt::detail::counting_buffer<>();
    fmt::detail::vformat_to(buf, ts, fmt.str, fmt::vargs<T...>{{args...}});
    return buf.count();
}
} // namespace detail

// Windows thinks fmtlib does heap bashing. I don't think it does, but it still causes segfaults.
template <typename... Args>
inline std::basic_string<char> format(fmt::runtime_format_string<> const &format, Args &&...args) {

    size_t out_size = fmt::formatted_size(format, std::forward<Args>(args)...);

    std::basic_string<char> out(out_size, 0);

    fmt::format_to(out.begin(), format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
inline std::basic_string<wchar_t> format(fmt::runtime_format_string<wchar_t> const &format, Args &&...args) {

    size_t out_size = fmt::formatted_size(format, std::forward<Args>(args)...);

    std::basic_string<wchar_t> out(out_size, 0);

    fmt::format_to(out.begin(), format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
std::basic_string<char> format(fmt::text_style style, fmt::runtime_format_string<> const &format, Args &&...args) {
    std::basic_string<char> out;

    size_t out_size = detail::formatted_size(style, format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), style, format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
std::basic_string<wchar_t> format(fmt::text_style style, fmt::runtime_format_string<wchar_t> const &format, Args &&...args) {
    std::basic_string<wchar_t> out;

    size_t out_size = detail::formatted_size(style, format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), style, format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
inline std::basic_string<char> format(std::basic_string_view<char> const &format, Args &&...args) {
    auto runtime_format = fmt::runtime(format);

    return einsums::format(runtime_format, std::forward<Args>(args)...);
}

template <typename... Args>
inline std::basic_string<wchar_t> format(std::basic_string_view<wchar_t> const &format, Args &&...args) {
    auto runtime_format = fmt::runtime(format);

    return einsums::format(runtime_format, std::forward<Args>(args)...);
}

template <typename... Args>
std::basic_string<char> format(fmt::text_style style, std::basic_string_view<char> const &format, Args &&...args) {
    auto runtime_format = fmt::runtime(format);

    return einsums::format(style, runtime_format, std::forward<Args>(args)...);
}

template <typename... Args>
std::basic_string<wchar_t> format(fmt::text_style style, std::basic_string_view<wchar_t> const &format, Args &&...args) {
    auto runtime_format = fmt::runtime(format);

    return einsums::format(style, runtime_format, std::forward<Args>(args)...);
}

} // namespace einsums
