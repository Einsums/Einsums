#pragma once

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

} // namespace print

namespace detail {
void println(const std::string &oss);
}

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
                format->leading_junk = i;
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
namespace detail {

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

} // namespace detail

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    detail::TuplePrinter<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple_no_type(const std::tuple<Args...> &t) -> std::string {
    std::ostringstream out;
    out << "(";
    detail::TuplePrinterNoType<decltype(t), sizeof...(Args)>::print(out, t);
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
void println(const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(format, ts...);
    detail::println(s);
}

template <typename... Ts>
void println(const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, format, ts...);
    detail::println(s);
}

inline void println(const std::string &format) {
    detail::println(format);
}

inline void println(const fmt::text_style &style, const std::string_view &format) {
    std::string s = fmt::format(style, format);
    detail::println(s);
}

inline void println() {
    detail::println("\n");
}

template <typename... Ts>
inline void println_abort(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    print::stacktrace();

    std::abort();
}

template <typename... Ts>
inline void println_warn(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);
}

FMT_BEGIN_NAMESPACE
template <class charT>
struct formatter<std::complex<double>, charT> {
  private:
    detail::dynamic_format_specs<char> specs_;

  public:
    FMT_CONSTEXPR typename basic_format_parse_context<charT>::iterator parse(basic_format_parse_context<charT> &ctx) {
        using handler_type = detail::dynamic_specs_handler<basic_format_parse_context<charT>>;
        detail::specs_checker<handler_type> handler(handler_type(specs_, ctx), detail::type::string_type);
        auto it = parse_format_specs(ctx.begin(), ctx.end(), handler);
        detail::parse_float_type_spec(specs_, ctx.error_handler());
        return it;
    }

    template <class FormatContext>
    typename FormatContext::iterator format(const std::complex<double> &c, FormatContext &ctx) {
        detail::handle_dynamic_spec<detail::precision_checker>(specs_.precision, specs_.precision_ref, ctx);
        auto specs = std::string();
        specs += "+";
        if (specs_.precision > 0)
            specs += fmt::format(".{}", specs_.precision);
        if (specs_.type == presentation_type::fixed_lower)
            specs += 'f';
        auto real = fmt::format(ctx.locale().template get<std::locale>(), fmt::runtime("{:" + specs + "}"), c.real());
        auto imag = fmt::format(ctx.locale().template get<std::locale>(), fmt::runtime("{:" + specs + "}"), c.imag());
        auto fill_align_width = std::string();
        if (specs_.width > 0)
            fill_align_width = fmt::format(">{}", specs_.width);
        return format_to(ctx.out(), runtime("{:" + fill_align_width + "}"), fmt::format("({} {}i)", real, imag));
    }
};

template <class charT>
struct formatter<std::complex<float>, charT> {
  private:
    detail::dynamic_format_specs<char> specs_;

  public:
    FMT_CONSTEXPR typename basic_format_parse_context<charT>::iterator parse(basic_format_parse_context<charT> &ctx) {
        using handler_type = detail::dynamic_specs_handler<basic_format_parse_context<charT>>;
        detail::specs_checker<handler_type> handler(handler_type(specs_, ctx), detail::type::string_type);
        auto it = parse_format_specs(ctx.begin(), ctx.end(), handler);
        detail::parse_float_type_spec(specs_, ctx.error_handler());
        return it;
    }

    template <class FormatContext>
    typename FormatContext::iterator format(const std::complex<float> &c, FormatContext &ctx) {
        detail::handle_dynamic_spec<detail::precision_checker>(specs_.precision, specs_.precision_ref, ctx);
        auto specs = std::string();
        specs += "+";
        if (specs_.precision > 0)
            specs += fmt::format(".{}", specs_.precision);
        if (specs_.type == presentation_type::fixed_lower)
            specs += 'f';
        auto real = fmt::format(ctx.locale().template get<std::locale>(), fmt::runtime("{:" + specs + "}"), c.real());
        auto imag = fmt::format(ctx.locale().template get<std::locale>(), fmt::runtime("{:" + specs + "}"), c.imag());
        auto fill_align_width = std::string();
        if (specs_.width > 0)
            fill_align_width = fmt::format(">{}", specs_.width);
        return format_to(ctx.out(), runtime("{:" + fill_align_width + "}"), fmt::format("({} {}i)", real, imag));
    }
};
FMT_END_NAMESPACE