/*
 * Copyright (c) 2022 Justin Turney
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file Print.hpp
 *
 * Contains printing functions.
 */

#pragma once

#include "einsums/_Export.hpp"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

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

/**
 * @struct Indent
 *
 * Represents an indentation. @todo Better description.
 */
struct Indent {
    Indent() { indent(); }
    ~Indent() { deindent(); }
};

/**
 * Print a stacktrace. @todo Double check the description.
 */
void EINSUMS_EXPORT stacktrace();

} // namespace print

namespace detail {
/**
 * Print a line to the output stream.
 *
 * @param oss The line to print.
 */
void EINSUMS_EXPORT println(const std::string &oss);
}

/**
 * Taken from https://stackoverflow.com/posts/59522794/revisions
 */
namespace detail {
template <typename T>
constexpr auto RawTypeName() -> const auto & {
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

/**
 * @struct RawTypeNameFormat
 *
 * @todo Describe this struct.
 */
struct RawTypeNameFormat {
    std::size_t leading_junk = 0, trailing_junk = 0;
};

/**
 * Gets a formatted type name string. @todo Double check.
 *
 * @param format The output parameter which will contain the format.
 *
 * @return True on success, false on failure.
 */
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

/**
 * @var format
 *
 * @todo Describe this.
 */
inline static constexpr RawTypeNameFormat format = [] {
    static_assert(GetRawTypeNameFormat(nullptr), "Unable to figure out how to generate type names on this compiler.");
    RawTypeNameFormat format;
    GetRawTypeNameFormat(&format);
    return format;
}();
} // namespace detail

// Returns the type name in a `std::array<char, N>` (null-terminated).
/**
 * Returns the type name of the value type in an array.
 * 
 * @return Type name of the value type in an array.
 */
template <typename T>
[[nodiscard]] constexpr auto CexprTypeName() {
    constexpr std::size_t len = sizeof(detail::RawTypeName<T>()) - detail::format.leading_junk - detail::format.trailing_junk;
    std::array<char, len> name{};
    for (std::size_t i = 0; i < len - 1; i++)
        name[i] = detail::RawTypeName<T>()[i + detail::format.leading_junk];
    return name;
}

/**
 * Returns the type name of an array as a string.
 *
 * @return A string representing the name of the type.
 */
template <typename T>
[[nodiscard]] auto type_name() -> const char * {
    static constexpr auto name = CexprTypeName<T>();
    return name.data();
}

/**
 * returns the type name of the input as a string.
 * 
 * @param The object whose type is to be returned.
 * 
 * @return A string representing the name of the type of the parameter.
 */
template <typename T>
[[nodiscard]] auto type_name(const T &) -> const char * {
    return type_name<T>();
}
namespace detail {

/**
 * @struct TuplePrinter
 *
 * Prints a tuple.
 */
template <typename Tuple, std::size_t N>
struct TuplePrinter {
    /**
     * Prints a tuple to the output stream.
     *
     * @param os Output stream.
     * @param t Tuple to print.
     */
    static void print(std::ostream &os, const Tuple &t) {
        TuplePrinter<Tuple, N - 1>::print(os, t);
        os << ", (" << type_name<decltype(std::get<N - 1>(t))>() << ")" << std::get<N - 1>(t);
    }
};

/**
 * @struct TuplePrinter
 * 
 * Base case for printing a tuple.
 */
template <typename Tuple>
struct TuplePrinter<Tuple, 1> {
    /**
     * Prints a tuple with one element.
     *
     * @param os The output stream.
     * @param t The tuple to print.
     */
    static void print(std::ostream &os, const Tuple &t) { os << "(" << type_name<decltype(std::get<0>(t))>() << ")" << std::get<0>(t); }
};

/**
 * @struct TuplePrinterNoType
 *
 * Print a tuple without type information.
 */
template <typename Tuple, std::size_t N>
struct TuplePrinterNoType {
    /**
     * Print a tuple without type information.
     *
     * @param os The output stream.
     * @param t The tuple to print.
     */
    static void print(std::ostream &os, const Tuple &t) {
        TuplePrinterNoType<Tuple, N - 1>::print(os, t);
        os << ", " << std::get<N - 1>(t);
    }
};

/**
 * @struct TuplePrinterNoType
 *
 * Base case for printing tuples without types.
 */
template <typename Tuple>
struct TuplePrinterNoType<Tuple, 1> {
    /**
     * Print a tuple with only one element without printing the type.
     *
     * @param os The output stream.
     * @param t The tuple to print.
     */
    static void print(std::ostream &os, const Tuple &t) { os << std::get<0>(t); }
};

/**
 * Prints an empty tuple.
 * 
 * @param The tuple to print.
 *
 * @return A string representing the tuple.
 */
template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple(const std::tuple<Args...> &) -> std::string {
    return {"()"};
}

/**
 * Prints an empty tuple without a type.
 *
 * @param The tuple to print. 
 *
 * @return A string representing the tuple.
 */
template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &) -> std::string {
    return {"()"};
}

} // namespace detail

/**
 * Print a tuple.
 *
 * @param t The tuple to print.
 *
 * @return A string representing the tuple.
 */
template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    detail::TuplePrinter<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

/**
 * Print a tuple without type.
 * 
 * @param t The tuple to print.
 *
 * @return A string representing the tuple.
 */
template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    detail::TuplePrinterNoType<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

/**
 * Print an empty tuple without type.
 *
 * @param The tuple to print.
 *
 * @return A string representing the tuple.
 */
inline auto print_tuple_no_type(const std::tuple<> &) -> std::string {
    std::ostringstream out;
    out << "( )";
    return out.str();
}

using fmt::bg;       // NOLINT
using fmt::color;    // NOLINT
using fmt::emphasis; // NOLINT
using fmt::fg;       // NOLINT

/**
 * Print a series of objects using the given format string.
 * 
 * @param f The format string.
 * @param ts The objects to print.
 */
template <typename... Ts>
void println(const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    detail::println(s);
}

/**
 * Print a series of objects using the given format string and style.
 *
 * @param style The style to use when printing.
 * @param format The format string.
 * @param ts The objects to print.
 */
template <typename... Ts>
void println(const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, format, ts...);
    detail::println(s);
}

/**
 * Print a format string with no formatted arguments.
 *
 * @param format The string to print.
 */
inline void println(const std::string &format) {
    detail::println(format);
}

/**
 * Print a format string with no formatted arguments in the given style.
 * 
 * @param style The style to print with.
 * @param format The format string to print.
 */
inline void println(const fmt::text_style &style, const std::string_view &format) {
    std::string s = fmt::format(style, format);
    detail::println(s);
}

/**
 * Print an empty line.
 */
inline void println() {
    detail::println("\n");
}

/**
 * Print a line then abort. Contains an error message.
 *
 * @param format The format string.
 * @param ts A list of objects to use for the format string.
 */
template <typename... Ts>
inline void println_abort(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    print::stacktrace();

    std::abort();
}

/**
 * Print a warning.
 * 
 * @param format The format string.
 * @param ts The objects to use for the format string.
 */
template <typename... Ts>
inline void println_warn(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);
}
