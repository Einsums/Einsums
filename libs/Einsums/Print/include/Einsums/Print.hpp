//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/File.hpp>

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

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

/**
 * @struct ordinal
 *
 * Wraps an integer. When the ordinal is printed, it will add the appropriate ordinal
 * suffix to that integer.
 */
template <std::integral IntType>
struct ordinal {
  public:
    constexpr ordinal() = default;

    /**
     * Copy constructor.
     */
    constexpr ordinal(ordinal<IntType> const &other) : val_{other.val_} {}

    /**
     * Move constructor.
     */
    constexpr ordinal(ordinal<IntType> &&other) : val_{std::move(other.val_)} {}

    /**
     * Cast constructor.
     */
    constexpr ordinal(IntType const &value) : val_{value} {}

    /**
     * Deleter.
     */
    constexpr ~ordinal() = default;

    /**
     * Copy assignment.
     */
    constexpr ordinal<IntType> &operator=(ordinal<IntType> const &other) {
        val_ = other.val_;
        return *this;
    }

    /**
     * Move assignment.
     */
    constexpr ordinal<IntType> &operator=(ordinal<IntType> &&other) {
        val_ = std::move(other.val_);
        return *this;
    }

    /**
     * Cast to underlying integer type.
     */
    constexpr operator IntType() const { return val_; }

    /**
     * Cast to reference of the underlying integer type.
     */
    constexpr operator IntType &() { return val_; }

    /**
     * Cast to arbitrary type.
     */
    template <typename T>
    constexpr operator T() const {
        return T(val_);
    }

#ifndef DOXYGEN
#    define OPERATOR(OP)                                                                                                                   \
        template <std::integral OtherType>                                                                                                 \
        constexpr ordinal<IntType> &operator OP##=(const ordinal<OtherType> &other) {                                                      \
            val_ OP## = other.val_;                                                                                                        \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
        template <std::integral OtherType>                                                                                                 \
        constexpr ordinal<IntType> &operator OP##=(const OtherType & other) {                                                              \
            val_ OP## = other;                                                                                                             \
            return *this;                                                                                                                  \
        }

    OPERATOR(+)
    OPERATOR(-)
    OPERATOR(*)
    OPERATOR(/)
    OPERATOR(%)
    OPERATOR(<<)
    OPERATOR(>>)
    OPERATOR(^)
    OPERATOR(&)
    OPERATOR(|)
#    undef OPERATOR

    template <std::integral OtherType>
    constexpr auto operator<=>(ordinal<OtherType> const &other) const {
        return val_ <=> other.val_;
    }

    template <std::integral OtherType>
    constexpr auto operator<=>(OtherType const &other) const {
        return val_ <=> other;
    }

    template <std::integral OtherType>
    constexpr bool operator==(ordinal<OtherType> const &other) const {
        return val_ == other.val_;
    }

    template <std::integral OtherType>
    constexpr bool operator==(OtherType const &other) const {
        return val_ == other;
    }
#endif

  private:
    /**
     * @var val_
     *
     * The value wrapped by the ordinal.
     */
    IntType val_{0};
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

#ifdef DOXYGEN
/**
 * Prints something to standard output. A new line is emmitted after the print is done.
 */
template <typename... Ts>
void println(Ts... args) {
    ;
}

/**
 * Prints something to a file pointer or output stream. A new line is emmitted after the print is done.
 */
template <typename OutType, typename... Ts>
void fprintln(OutType out, Ts... args) {
    ;
}
#endif

#ifndef DOXYGEN
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
#endif

/**
 * Calls println to generate an error message, then aborts.
 */

template <typename... Ts>
[[deprecated("Raise an exception instead. The EINSUMS_THROW_EXCEPTION macro provides way more information on what went wrong than this "
             "function.")]] void
println_abort(std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(color::red) | fg(color::white), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace().print();
#endif

    std::abort();
}

/**
 * Calls println to generate a warning message.
 */
template <typename... Ts>
[[deprecated("Use our logging functionality. EINSUMS_LOG_WARN provides way more information than this.")]]
void println_warn(std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(color::yellow) | fg(color::black), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace(0, 3).print();
#endif
}

/**
 * Calls fprintln to generate an error message, then aborts.
 */
template <typename... Ts>
void fprintln_abort(std::FILE *fp, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(fp, message, ts...);

    std::abort();
}

/**
 * Calls fprintln to generate a warning message.
 */
template <typename... Ts>
void fprintln_warn(std::FILE *fp, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(fp, message, ts...);
}

#ifndef DOXYGEN
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
#endif

} // namespace einsums

#if !defined(DOXYGEN)

template <std::integral IntType>
struct fmt::formatter<einsums::print::ordinal<IntType>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return fmt_.parse(ctx);
    }

    template <typename FormatContext>
    auto format(einsums::print::ordinal<IntType> const &value, FormatContext &ctx) const {
        ctx.advance_to(fmt_.format((IntType)value, ctx));
        auto suffix = get_suffix(value);
        return fmt::format_to(ctx.out(), "{}", suffix);
    }

  protected:
    constexpr inline char const *get_suffix(IntType value) const {
        IntType const hundreds = (value % 100 + 100) % 100;

        if (hundreds > 20 || hundreds < 10) {
            switch (hundreds % 10) {
            case 1:
                return "st";
            case 2:
                return "nd";
            case 3:
                return "rd";
            default:
                return "th";
            }
        } else {
            return "th";
        }
    }

  private:
    fmt::formatter<IntType> fmt_;
};

#endif