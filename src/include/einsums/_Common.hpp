#pragma once

#include "einsums/Error.hpp"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_Export.hpp"

#include <array>
#include <cstdint>
#include <ostream>

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
    extern EINSUMS_EXPORT std::string s_Namespace;                                                                                         \
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
    EINSUMS_EXPORT std::string s_Namespace = #x;                                                                                           \
    }

/**
 * Matching macro to BEGIN_EINSUMS_NAMESPACE_HPP(x)
 */
#define END_EINSUMS_NAMESPACE_CPP(x) }

namespace einsums {

#if defined(MKL_ILP64)
using eint  = long long int;
using euint = unsigned long long int;
using elong = long long int;
#else
using eint  = int;
using euint = unsigned int;
using elong = long int;
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
 * Shuts down Einsums and possibly print out a timings report.
 *
 * @param timerReport whether to print the timings report of not. Defaults to false.
 */
void EINSUMS_EXPORT finalize(bool timerReport = false);

// The following detail and "using" statements below are needed to ensure Dims, Strides, and Offsets are strong-types in C++
namespace detail {

struct DimType {};
struct StrideType {};
struct OffsetType {};
struct CountType {};
struct RangeType {};
struct ChunkType {};

template <typename T, std::size_t Rank, typename UnderlyingType = std::size_t>
struct Array : public std::array<UnderlyingType, Rank> {
    template <typename... Args>
    constexpr explicit Array(Args... args) : std::array<UnderlyingType, Rank>{static_cast<UnderlyingType>(args)...} {}
    using Type = T;
};
} // namespace detail

template <std::size_t Rank>
using Dim = detail::Array<detail::DimType, Rank, std::int64_t>;

template <std::size_t Rank>
using Stride = detail::Array<detail::StrideType, Rank>;

template <std::size_t Rank>
using Offset = detail::Array<detail::OffsetType, Rank>;

template <std::size_t Rank>
using Count = detail::Array<detail::CountType, Rank>;

using Range = detail::Array<detail::RangeType, 2, std::int64_t>;

template <std::size_t Rank>
using Chunk = detail::Array<detail::ChunkType, Rank, std::int64_t>;

struct All_t {};
static struct All_t All;

} // namespace einsums

/**
 * Function for printing Dim object.
 */
template <size_t Rank>
void println(const einsums::Dim<Rank> &dim) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << dim[i] << " ";
    }
    println("Dim{{{}}}", oss.str());
}

template <size_t Rank>
struct fmt::formatter<einsums::Dim<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw_format_error("invalid format");

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

/**
 * Function for printing Stride object.
 */
template <size_t Rank>
void println(const einsums::Stride<Rank> &stride) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << stride[i] << " ";
    }
    println("Stride{{{}}}", oss.str().c_str());
}

template <size_t Rank>
struct fmt::formatter<einsums::Stride<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw_format_error("invalid format");

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

/**
 * Function for printing Count object.
 */
template <size_t Rank>
void println(const einsums::Count<Rank> &count) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << count[i] << " ";
    }
    println("Count{{{}}}", oss.str().c_str());
}

template <size_t Rank>
struct fmt::formatter<einsums::Count<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw_format_error("invalid format");

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

/**
 * Function for printing Offset object.
 */
template <size_t Rank>
void println(const einsums::Offset<Rank> &offset) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << offset[i] << " ";
    }
    println("Offset{{{}}}", oss.str().c_str());
}

template <size_t Rank>
struct fmt::formatter<einsums::Offset<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw_format_error("invalid format");

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

/**
 * Function for printing Range object.
 */
inline void println(const einsums::Range &range) {
    std::ostringstream oss;
    oss << range[0] << " " << range[1];
    println("Range{{{}}}", oss.str().c_str());
}

template <>
struct fmt::formatter<einsums::Range> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            throw_format_error("invalid format");

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

/**
 * Function for printing std::array object.
 */
template <size_t Rank, typename T>
inline void println(const std::array<T, Rank> &array) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << array[i] << " ";
    }
    println("std::array{{{}}}", einsums::rtrim_copy(oss.str().c_str()));
}

// Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define ALIAS_TEMPLATE_FUNCTION(highLevelFunction, lowLevelFunction)                                                                       \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args)->decltype(lowLevelFunction(std::forward<Args>(args)...)) {                               \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }
