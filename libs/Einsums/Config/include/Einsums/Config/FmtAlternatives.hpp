//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/*
 * Windows doesn't like certain library functions. We want to use library functions. This header provides wrappers for library functions
 * that don't throw errors on Windows.
 */

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/xchar.h>

#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <string>
#include <type_traits>

#ifdef EINSUMS_WINDOWS
#    include <process.h>
#    include <stdlib.h>
#else
#    include <unistd.h>
#endif

namespace einsums {
namespace detail {

// fmtlib doesn't have formatted_size for styled text. I'll just make my own.
template <typename... T>
FMT_NODISCARD FMT_INLINE auto formatted_size(fmt::text_style ts, fmt::format_string<T...> fmt, T &&...args) -> size_t {
    auto buf = fmt::detail::counting_buffer<>();
    fmt::detail::vformat_to(buf, ts, fmt.str, fmt::vargs<T...>{{args...}});
    return buf.count();
}

// Windows thinks fmtlib does heap bashing. I don't think it does, but it still causes segfaults.
template <typename... Args>
inline std::basic_string<char> corrected_format(std::basic_string_view<char> const &format, Args &&...args) {
    auto runtime_format = fmt::runtime(format);

    size_t out_size = fmt::formatted_size(runtime_format, std::forward<Args>(args)...);

    std::basic_string<char> out(out_size, 0);

    fmt::format_to(out.begin(), runtime_format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
inline std::basic_string<wchar_t> corrected_format(std::basic_string_view<wchar_t> const &format, Args &&...args) {

    auto runtime_format = fmt::runtime(format);

    size_t out_size = fmt::formatted_size(runtime_format, std::forward<Args>(args)...);

    std::basic_string<wchar_t> out(out_size, 0);

    fmt::format_to(out.begin(), runtime_format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
std::basic_string<char> corrected_format(fmt::text_style style, std::basic_string_view<char> const &format, Args &&...args) {
    std::basic_string<char> out;

    auto runtime_format = fmt::runtime(format);

    size_t out_size = detail::formatted_size(style, runtime_format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), style, runtime_format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
std::basic_string<wchar_t> corrected_format(fmt::text_style style, std::basic_string_view<wchar_t> const &format, Args &&...args) {
    std::basic_string<wchar_t> out;

    auto runtime_format = fmt::runtime(format);

    size_t out_size = detail::formatted_size(style, runtime_format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), style, runtime_format, std::forward<Args>(args)...);

    return out;
}
} // namespace detail
} // namespace einsums
