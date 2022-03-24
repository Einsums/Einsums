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

namespace einsums {

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

void stacktrace();

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
    prefix = "auto einsums::type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto einsums::type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl einsums::type_name<";
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

inline auto print_tuple_no_type(const std::tuple<> &) -> std::string {
    std::ostringstream out;
    out << "( )";
    return out.str();
}

} // namespace einsums

namespace einsums {

using fmt::bg;       // NOLINT
using fmt::color;    // NOLINT
using fmt::emphasis; // NOLINT
using fmt::fg;       // NOLINT

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

template <typename... Ts>
inline void println_abort(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    Print::stacktrace();

    std::abort();
}

} // namespace einsums