//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Export.hpp"

#include "einsums/STL.hpp"

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <sstream>

#define EINSUMS_STRINGIFY(a)  EINSUMS_STRINGIFY2(a)
#define EINSUMS_STRINGIFY2(a) #a

/**
 * Handles setting up the namespace and creates a global std::string that is used
 * by the timing mechanism to track locations.
 *
 * @param x The name of the namespace to create and store the name of.
 *
 */
#define BEGIN_EINSUMS_NAMESPACE_HPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace detail {                                                                                                                     \
    EINSUMS_EXPORT const std::string &get_namespace();                                                                                     \
    }

/**
 * The matching macro for BEGIN_EINSUMS_NAMESPACE_HPP(x)
 */
#define END_EINSUMS_NAMESPACE_HPP(x) }

/**
 * The .cpp file equivalent of BEGIN_EINSUMS_NAMESPACE_HPP. Should only exist in one
 * source file; otherwise, multiple definition errors will occur.
 */
#define BEGIN_EINSUMS_NAMESPACE_CPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace detail {                                                                                                                     \
    namespace {                                                                                                                            \
    std::string s_Namespace = #x;                                                                                                          \
    }                                                                                                                                      \
    EINSUMS_EXPORT const std::string &get_namespace() {                                                                                    \
        return s_Namespace;                                                                                                                \
    }                                                                                                                                      \
    }

#if !defined(EINSUMS_ZERO)
#    define EINSUMS_ZERO (1e-10)
#endif

/**
 * Matching macro to BEGIN_EINSUMS_NAMESPACE_HPP(x)
 */
#define END_EINSUMS_NAMESPACE_CPP(x) }

namespace einsums {

#if EINSUMS_BLAS_INTERFACE_ILP64
using blas_int   = long long int;          // NOLINT
using blas_euint = unsigned long long int; // NOLINT
using blas_elong = long long int;          // NOLINT
#elif EINSUMS_BLAS_INTERFACE_LP64
using blas_int   = int;          // NOLINT
using blas_euint = unsigned int; // NOLINT
using blas_elong = long int;     // NOLINT
#else
#    warning Unknown BLAS interface type. Defaulting to LP64
using blas_int   = int;          // NOLINT
using blas_euint = unsigned int; // NOLINT
using blas_elong = long int;     // NOLINT
#endif

/**
 * @brief Handles initializing the internals of Einsums.
 *
 * The current implementation initializes the timer system, calls
 * on the blas subsystem to initialize itself (for example, gpu variant would
 * obtain global device handle), prevents OpenMP from allowing nested
 * OpenMP regions (leading to oversubscription), and disables HDF5
 * diagnostic reporting.
 *
 * In a future parallel variant of Einsums, this would also initialize
 * the MPI runtime.
 *
 * @return int on success returns 0, on failure anything else.
 */
auto EINSUMS_EXPORT initialize() -> int;

/**
 * Shuts down Einsums and prints out the timing report to the specified file.
 */
EINSUMS_EXPORT void finalize(const char *output_file);
EINSUMS_EXPORT void finalize(const std::string &output_file);
EINSUMS_EXPORT void finalize(FILE *file_pointer);
EINSUMS_EXPORT void finalize(std::ostream &output_stream);

/**
 * Shuts down Einsums and possibly print out a timings report to stdout.
 *
 * @param timerReport whether to print the timings report of not. Defaults to false.
 */
EINSUMS_EXPORT void finalize(bool timerReport = false);

#define DEFINE_STRUCT(Name, UnderlyingType)                                                                                                \
    template <std::size_t Rank>                                                                                                            \
    struct Name : std::array<std::int64_t, Rank> {                                                                                         \
        template <typename... Args>                                                                                                        \
        constexpr explicit Name(Args... args) : std::array<std::int64_t, Rank>{static_cast<std::int64_t>(args)...} {                       \
        }                                                                                                                                  \
    };                                                                                                                                     \
    template <typename... Args>                                                                                                            \
    Name(Args... args)->Name<sizeof...(Args)> /**/

DEFINE_STRUCT(Dim, std::int64_t);
DEFINE_STRUCT(Stride, std::size_t);
DEFINE_STRUCT(Offset, std::size_t);
DEFINE_STRUCT(Count, std::size_t);
DEFINE_STRUCT(Chunk, std::int64_t);

struct Range : std::array<std::int64_t, 2> {
    template <typename... Args>
    constexpr explicit Range(Args... args) : std::array<std::int64_t, 2>{static_cast<std::int64_t>(args)...} {}
};

struct AllT {};
static struct AllT All; // NOLINT

} // namespace einsums

template <size_t Rank>
struct fmt::formatter<einsums::Dim<Rank>> {
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
    auto format(const einsums::Dim<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Dim{{{}}}", einsums::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct fmt::formatter<einsums::Stride<Rank>> {
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
    auto format(const einsums::Stride<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Stride{{{}}}", einsums::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct fmt::formatter<einsums::Count<Rank>> {
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
    auto format(const einsums::Count<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Count{{{}}}", einsums::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct fmt::formatter<einsums::Offset<Rank>> {
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
    auto format(const einsums::Offset<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Offset{{{}}}", einsums::rtrim_copy(oss.str()));
    }
};

template <>
struct fmt::formatter<einsums::Range> {
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
    auto format(const einsums::Range &dim, format_context &ctx) const -> format_context::iterator {
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Range{{{}, {}}}", dim[0], dim[1]);
    }
};

// Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define ALIAS_TEMPLATE_FUNCTION(highLevelFunction, lowLevelFunction)                                                                       \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args) -> decltype(lowLevelFunction(std::forward<Args>(args)...)) {                             \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }
