//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <range/v3/utility/common_tuple.hpp>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <iostream>
#include <string_view>

namespace einsums {
namespace print {

/// Add spaces to the global indentation counter.
void EINSUMS_EXPORT indent();

/// Removes spaces from the global indentation counter.
void EINSUMS_EXPORT deindent();

/// Returns the current indentation level.
auto EINSUMS_EXPORT current_indent_level() -> int;

/**
 * @brief Controls whether a line header is printed for the main thread or not.
 *
 * @param onoff If true, print thread id for main and child threads, otherwise just print for child threads.
 */
void EINSUMS_EXPORT always_print_thread_id(bool onoff);

/**
 * @brief Silences all output.
 *
 * @param onoff If true, output is suppressed, otherwise printing is allowed.
 */
void EINSUMS_EXPORT suppress_output(bool onoff);

/**
 * @brief Controls indentation.
 */
struct Indent {
     Indent() { indent(); }
    ~Indent() { deindent(); }
};

} // namespace print

/// \cond NOINTERNAL
namespace detail {
void EINSUMS_EXPORT println(std::string const &str);
void EINSUMS_EXPORT fprintln(std::FILE *fp, std::string const &str);
void EINSUMS_EXPORT fprintln(std::ostream &os, std::string const &str);
} // namespace detail
/// \endcond NOINTERNAL

using fmt::bg;
using fmt::color;
using fmt::emphasis;
using fmt::fg;

template <typename... Ts>
void println(std::string_view const &f, Ts const... ts) {
    std::string const s = fmt::format(fmt::runtime(f), ts...);
    detail::println(s);
}

template <typename... Ts>
void println(fmt::text_style const &style, std::string_view const &format, Ts const... ts) {
    std::string const s = fmt::format(style, fmt::runtime(format), ts...);
    detail::println(s);
}

inline void println(fmt::text_style const &style, std::string_view const &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    detail::println(s);
}

inline void println() {
    detail::println("\n");
}

template <typename... Ts>
void fprintln(std::FILE *fp, std::string_view const &f, Ts const... ts) {
    std::string const s = fmt::format(fmt::runtime(f), ts...);
    detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::FILE *fp, fmt::text_style const &style, std::string_view const &format, Ts const... ts) {
    std::string s;
    if (fp == stdout || fp == stderr) {
        s = fmt::format(style, format, ts...);
    } else {
        s = fmt::format(format, ts...);
    }
    detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp, std::string const &format) {
    detail::fprintln(fp, format);
}

inline void fprintln(std::FILE *fp, fmt::text_style const &style, std::string_view const &format) {
    std::string s;
    if (fp == stdout || fp == stderr) {
        s = fmt::format(style, fmt::runtime(format));
    } else {
        s = format;
    }
    detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp) {
    detail::fprintln(fp, "\n");
}

template <typename... Ts>
void fprintln(std::ostream &fp, std::string_view const &f, Ts const... ts) {
    std::string const s = fmt::format(fmt::runtime(f), ts...);
    detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::ostream &fp, fmt::text_style const &style, std::string_view const &format, Ts const... ts) {
    std::string const s = fmt::format(style, format, ts...);
    detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp, std::string const &format) {
    detail::fprintln(fp, format);
}

inline void fprintln(std::ostream &fp, fmt::text_style const &style, std::string_view const &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp) {
    detail::fprintln(fp, "\n");
}

template <typename... Ts>
void println_abort(std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(color::red) | fg(color::white), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace().print();
#endif

    std::abort();
}

template <typename... Ts>
void println_warn(std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(color::yellow) | fg(color::black), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace(0, 3).print();
#endif
}

template <typename... Ts>
void fprintln_abort(std::FILE *fp, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(fp, message, ts...);

    std::abort();
}

template <typename... Ts>
void fprintln_warn(std::FILE *fp, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(fp, message, ts...);
}

template <typename... Ts>
void fprintln_abort(std::ostream &os, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(os, bg(color::red) | fg(color::white), message, ts...);

    std::abort();
}

template <typename... Ts>
void fprintln_warn(std::ostream &os, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(os, bg(color::yellow) | fg(color::black), message, ts...);
}

} // namespace einsums

#if !defined(DOXYGEN)
template <typename... Ts>
struct fmt::formatter<ranges::common_tuple<Ts...>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(ranges::common_tuple<Ts...> const &ct, FormatContext &ctx) {
        // Create a tuple from the common_tuple
        auto tpl = static_cast<std::tuple<Ts...>>(ct);

        // Join the tuple elements with a separator (default ", ")
        return fmt::format_to(ctx.out(), "{}", fmt::join(tpl, ", "));
    }
};
#endif
