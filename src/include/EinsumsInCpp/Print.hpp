#pragma once

#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EinsumsInCpp {

// Forward declare the Tensor object for printing purposes
template <size_t Rank, typename T>
struct Tensor;

namespace Print {

/** Adds spaces to the global indentation counter. */
void indent();
/** Removes spaces from the global indentation counter. */
void deindent();

/** Returns the current indentation level. */
auto current_indent_level() -> int;

/**
 * @brief Controls whether a line header is printed for the main thread or not.
 *
 * @param onoff If true, print thread id for main and child threads, otherwise just print for child threads.
 *
 *
 */
void always_print_thread_id(bool onoff);

/**
 * @brief Silences all output.
 *
 * @param onoff If true, output is suppressed, otherwise printing is allowed.
 */
void suppress_output(bool onoff);

struct Indent {
    Indent() { indent(); }
    ~Indent() { deindent(); }
};

} // namespace Print

namespace Detail {
void println(const std::string &oss);
}

//
// Taken from https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c/56766138#56766138
//
template <typename T>
constexpr auto type_name() noexcept {
    std::string_view name = "Error: unsupported compiler", prefix, suffix;

#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto EinsumsInCpp::type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto EinsumsInCpp::type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl EinsumsInCpp::type_name<";
    suffix = ">(void) noexcept";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

namespace Detail {

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

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple(const std::tuple<Args...> &) -> std::string {
    return {"()"};
}

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &) -> std::string {
    return {"()"};
}

} // namespace Detail

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    Detail::TuplePrinter<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    Detail::TuplePrinterNoType<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

inline auto print_tuple_no_type(const std::tuple<> &t) -> std::string {
    std::ostringstream out;
    out << "( )";
    return out.str();
}

} // namespace EinsumsInCpp

namespace fmt {

// template <typename... Args>
// struct formatter<std::tuple<Args...>> {
//     // Presentation format: 't' - show types, 'n' - don't show types
//     char presentation = 'n';

//     // Parses format specification of the form ['t', 'n'].
//     constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
//         // [ctx.begin(), ctx.end()) is a character range that contains a part of
//         // the format string starting from the format specifications to be parsed,
//         // e.g. in
//         //
//         //   fmt::format("{:f} - point of interest", point{1, 2});
//         //
//         // the range will contain "f} - point of interest". The formatter should
//         // parse specifiers until '}' or the end of the range. In this example
//         // the formatter should parse the 'f' specifier and return an iterator
//         // pointing to '}'.

//         // Parse the presentation format and store it in the formatter
//         auto it = ctx.begin(), end = ctx.end();
//         if (it != end && (*it == 't' || *it == 'n'))
//             presentation = *it++;

//         // Check if reached the end of the range:
//         if (it != end && *it != '}')
//             throw format_error("invalid format");

//         return it;
//     }

//     // Formats the tuple using the parsed format specification (presentation)
//     template <typename FormatContext>
//     auto format(const std::tuple<Args...> &t, FormatContext &ctx) -> decltype(ctx.out()) {
//         return format_to(ctx.out(), "{:}", presentation == 'n' ? ::EinsumsInCpp::Detail::print_tuple_no_type(t) :
//         ::EinsumsInCpp::Detail::print_tuple(t));
//     }
// };

} // namespace fmt

namespace EinsumsInCpp {

template <typename... Ts>
void println(const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(format, ts...);
    Detail::println(s);
}

template <typename... Ts>
void println(const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, format, ts...);
    Detail::println(s);
}

inline void println(const std::string &format) { Detail::println(format); }

inline void println(const fmt::text_style &style, const std::string_view &format) {
    std::string s = fmt::format(style, format);
    Detail::println(s);
}

inline void println() { Detail::println("\n"); }

using fmt::bg;       // NOLINT
using fmt::color;    // NOLINT
using fmt::emphasis; // NOLINT
using fmt::fg;       // NOLINT

} // namespace EinsumsInCpp