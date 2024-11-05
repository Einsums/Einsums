//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string_view>
#include <tuple>

namespace print {

/** Adds spaces to the global indentation counter. */
void EINSUMS_EXPORT indent();
/** Removes spaces from the global indentation counter. */
void EINSUMS_EXPORT deindent();

/** Returns the current indentation level. */
auto EINSUMS_EXPORT current_indent_level() -> int;

/**
 * @brief Controls whether a line header is printed for the main thread or not.
 *
 * @param onoff If true, print thread id for main and child threads, otherwise just print for child threads.
 *
 *
 */
void EINSUMS_EXPORT always_print_thread_id(bool onoff);

/**
 * @brief Silences all output.
 *
 * @param onoff If true, output is suppressed, otherwise printing is allowed.
 */
void EINSUMS_EXPORT suppress_output(bool onoff);

struct Indent {
    Indent() { indent(); }
    ~Indent() { deindent(); }
};

} // namespace print

namespace einsums::detail {
void EINSUMS_EXPORT println(const std::string &oss);
void EINSUMS_EXPORT fprintln(std::FILE *fp, const std::string &oss);
void EINSUMS_EXPORT fprintln(std::ostream &os, const std::string &oss);
} // namespace einsums::detail

//
// Taken from https://stackoverflow.com/posts/59522794/revisions
//
namespace detail {
template <typename T>
constexpr auto RawTypeName() -> const auto & {
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

struct RawTypeNameFormat {
    std::size_t leading_junk = 0, trailing_junk = 0;
};

// Returns `false` on failure.
inline constexpr auto GetRawTypeNameFormat(RawTypeNameFormat *format) -> bool {
    const auto &str = RawTypeName<int>();
    for (std::size_t i = 0;; i++) {
        if (str[i] == 'i' && str[i + 1] == 'n' && str[i + 2] == 't') {
            if (format) {
                format->leading_junk  = i;
                format->trailing_junk = sizeof(str) - i - 3 - 1; // `3` is the length of "int", `1` is the space for the null terminator.
            }
            return true;
        }
    }
    return false;
}

inline static constexpr RawTypeNameFormat format = [] {
    static_assert(GetRawTypeNameFormat(nullptr), "Unable to figure out how to generate type names on this compiler.");
    RawTypeNameFormat format;
    GetRawTypeNameFormat(&format);
    return format;
}();
} // namespace detail

// Returns the type name in a `std::array<char, N>` (null-terminated).
template <typename T>
[[nodiscard]] constexpr auto CexprTypeName() {
    constexpr std::size_t len = sizeof(detail::RawTypeName<T>()) - detail::format.leading_junk - detail::format.trailing_junk;
    std::array<char, len> name{};
    for (std::size_t i = 0; i < len - 1; i++)
        name[i] = detail::RawTypeName<T>()[i + detail::format.leading_junk];
    return name;
}

template <typename T>
[[nodiscard]] auto type_name() -> const char * {
    static constexpr auto name = CexprTypeName<T>();
    return name.data();
}
template <typename T>
[[nodiscard]] auto type_name(const T &) -> const char * {
    return type_name<T>();
}

namespace einsums::detail {

template <typename Tuple, std::size_t N>
struct TuplePrinter {
    static void print(std::ostream &os, const Tuple &t) {
        TuplePrinter<Tuple, N - 1>::print(os, t);
        os << ", (" << type_name<decltype(std::get<N - 1>(t))>() << ")" << std::get<N - 1>(t);
    }
};

template <typename Tuple>
struct TuplePrinter<Tuple, 1> {
    static void print(std::ostream &os, const Tuple &t) { os << "(" << type_name<decltype(std::get<0>(t))>() << ")" << std::get<0>(t); }
};

template <typename Tuple, std::size_t N>
struct TuplePrinterNoType {
    static void print(std::ostream &os, const Tuple &t) {
        TuplePrinterNoType<Tuple, N - 1>::print(os, t);
        os << ", " << std::get<N - 1>(t);
    }
};

template <typename Tuple>
struct TuplePrinterNoType<Tuple, 1> {
    static void print(std::ostream &os, const Tuple &t) { os << std::get<0>(t); }
};

template <typename... Args>
    requires(sizeof...(Args) > 0)
auto print_tuple(const std::tuple<Args...> &) -> std::string {
    return {"()"};
}

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &) -> std::string {
    return {"()"};
}

} // namespace einsums::detail

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    ::einsums::detail::TuplePrinter<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    ::einsums::detail::TuplePrinterNoType<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

inline auto print_tuple_no_type(const std::tuple<> &) -> std::string {
    std::ostringstream out;
    out << "( )";
    return out.str();
}

using fmt::bg;       // NOLINT
using fmt::color;    // NOLINT
using fmt::emphasis; // NOLINT
using fmt::fg;       // NOLINT

template <typename... Ts>
void println(const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    ::einsums::detail::println(s);
}

template <typename... Ts>
void println(const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, fmt::runtime(format), ts...);
    ::einsums::detail::println(s);
}

inline void println(const std::string &format) {
    ::einsums::detail::println(format);
}

inline void println(const fmt::text_style &style, const std::string_view &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    ::einsums::detail::println(s);
}

inline void println() {
    ::einsums::detail::println("\n");
}

template <typename... Ts>
void fprintln(std::FILE *fp, const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    ::einsums::detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::FILE *fp, const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s;
    if (fp == stdout || fp == stderr) {
        s = fmt::format(style, format, ts...);
    } else {
        s = fmt::format(format, ts...);
    }
    ::einsums::detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp, const std::string &format) {
    ::einsums::detail::fprintln(fp, format);
}

inline void fprintln(std::FILE *fp, const fmt::text_style &style, const std::string_view &format) {
    std::string s;
    if (fp == stdout || fp == stderr) {
        s = fmt::format(style, fmt::runtime(format));
    } else {
        s = format;
    }
    ::einsums::detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp) {
    ::einsums::detail::fprintln(fp, "\n");
}

template <typename... Ts>
void fprintln(std::ostream &fp, const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    ::einsums::detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::ostream &fp, const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, format, ts...);
    ::einsums::detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp, const std::string &format) {
    ::einsums::detail::fprintln(fp, format);
}

inline void fprintln(std::ostream &fp, const fmt::text_style &style, const std::string_view &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    ::einsums::detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp) {
    ::einsums::detail::fprintln(fp, "\n");
}

template <typename... Ts>
inline void println_abort(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    std::abort();
}

template <typename... Ts>
inline void println_warn(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);
}

template <typename... Ts>
inline void fprintln_abort(std::FILE *fp, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(fp, message, ts...);

    std::abort();
}

template <typename... Ts>
inline void fprintln_warn(std::FILE *fp, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(fp, message, ts...);
}

template <typename... Ts>
inline void fprintln_abort(std::ostream &os, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(os, bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    std::abort();
}

template <typename... Ts>
inline void fprintln_warn(std::ostream &os, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(os, bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);
}
