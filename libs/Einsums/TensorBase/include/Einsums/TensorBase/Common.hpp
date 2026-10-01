//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/StringUtil/Trim.hpp>

#include <fmt/format.h>

#include <array>
#include <concepts>
#include <sstream>
#include <type_traits>

namespace einsums {

/**
 * @def DEFINE_STRUCT
 *
 * @brief Convenience macro for creating a type derived from a std::array.
 */
#define DEFINE_STRUCT(Name, UnderlyingType)                                                                                                \
    template <std::size_t Rank>                                                                                                            \
    struct Name : std::array<std::int64_t, Rank> {                                                                                         \
        template <typename... Args>                                                                                                        \
        constexpr explicit Name(Args... args) : std::array<std::int64_t, Rank>{static_cast<std::int64_t>(args)...} {                       \
        }                                                                                                                                  \
    };                                                                                                                                     \
    template <typename... Args>                                                                                                            \
    Name(Args... args)->Name<sizeof...(Args)>

/**
 * @struct Dim
 *
 * @brief Holds a list of dimensions in an array.
 */
template <std ::size_t Rank>
struct Dim : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     */
    template <typename... Args>
        requires(std::is_integral_v<std::remove_cvref_t<Args>> && ...)
    constexpr explicit Dim(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}

    template <typename Iterator>
        requires requires(Iterator it) {
            { *it };
            { it++ };
        }
    constexpr Dim(Iterator start, Iterator end) {
        auto this_it  = this->begin();
        auto other_it = start;

        while (this_it != this->end() && other_it != end) {
            *this_it = static_cast<std::int64_t>(*other_it);
            this_it++;
            other_it++;
        }
    }
};

/**
 * @struct Stride
 *
 * @brief Holds a list of strides in an array.
 */
template <std ::size_t Rank>
struct Stride : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     */
    template <typename... Args>
        requires(std::is_integral_v<std::remove_cvref_t<Args>> && ...)
    constexpr explicit Stride(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}

    template <typename Iterator>
        requires requires(Iterator it) {
            { *it };
            { it++ };
        }
    constexpr Stride(Iterator start, Iterator end) {
        auto this_it  = this->begin();
        auto other_it = start;

        while (this_it != this->end() && other_it != end) {
            *this_it = static_cast<std::int64_t>(*other_it);
            this_it++;
            other_it++;
        }
    }
};

/**
 * @struct Offset
 *
 * @brief Holds a list of offsets in an array.
 */
template <std ::size_t Rank>
struct Offset : std ::array<std ::int64_t, Rank> {
    using std::array<std::int64_t, Rank>::array;
    /**
     * @brief Aggregate constructor.
     */
    template <typename... Args>
        requires(std::is_integral_v<std::remove_cvref_t<Args>> && ...)
    constexpr explicit Offset(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}

    template <typename Iterator>
        requires requires(Iterator it) {
            { *it };
            { it++ };
        }
    constexpr Offset(Iterator start, Iterator end) {
        auto this_it  = this->begin();
        auto other_it = start;

        while (this_it != this->end() && other_it != end) {
            *this_it = static_cast<std::int64_t>(*other_it);
            this_it++;
            other_it++;
        }
    }
};

/**
 * @struct Count
 *
 * @brief Holds a list of counts in an array.
 */
template <std ::size_t Rank>
struct Count : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     */
    template <typename... Args>
    constexpr explicit Count(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}
};

/**
 * @struct Chunk
 *
 * @brief Holds a list of chunks in an array.
 */
template <std ::size_t Rank>
struct Chunk : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     */
    template <typename... Args>
    constexpr explicit Chunk(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}
};

#ifndef DOXYGEN
template <typename... Args>
Dim(Args... args) -> Dim<sizeof...(Args)>;

template <typename... Args>
Stride(Args... args) -> Stride<sizeof...(Args)>;

template <typename... Args>
Offset(Args... args) -> Offset<sizeof...(Args)>;

template <typename... Args>
Count(Args... args) -> Count<sizeof...(Args)>;

template <typename... Args>
Chunk(Args... args) -> Chunk<sizeof...(Args)>;
#endif

/**
 * @struct Range
 *
 * Holds two values: a starting value and an ending value.
 */
struct Range : std::array<std::int64_t, 2> {
    constexpr Range() = default;

    /**
     * Initialize a range.
     */
    template <std::integral First, std::integral Second>
    constexpr explicit Range(First first, Second second)
        : std::array<std::int64_t, 2>{static_cast<std::int64_t>(first), static_cast<std::int64_t>(second)} {}

    [[nodiscard]] bool is_removable() const noexcept { return _is_removable; }

  protected:
    bool _is_removable{false};
};

/**
 * @struct Range
 *
 * Holds two values: a starting value and an ending value. It will be treated as a single value if the start and end are the same.
 */
struct RemovableRange : Range {
    /**
     * Initialize a range.
     */
    template <std::integral First, std::integral Second>
    constexpr explicit RemovableRange(First first, Second second) : Range{first, second} {
        this->_is_removable = true;
    }
};

struct AllT {};
static struct AllT All; // NOLINT

#undef DEFINE_STRUCT

} // namespace einsums

namespace fmt {
template <size_t Rank>
struct formatter<einsums::Dim<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(einsums::Dim<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;

        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Dim{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

#if !defined(DOXYGEN)

template <size_t Rank>
struct formatter<einsums::Stride<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(einsums::Stride<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Stride{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct formatter<einsums::Count<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(einsums::Count<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Count{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct formatter<einsums::Offset<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(einsums::Offset<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Offset{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <>
struct formatter<einsums::Range> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(einsums::Range const &dim, format_context &ctx) const -> format_context::iterator {
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Range{{{}, {}}}", dim[0], dim[1]);
    }
};
} // namespace fmt
#endif
