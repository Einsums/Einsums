#pragma once

#include "einsums/Print.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_Export.hpp"

#include <array>
#include <cstdint>
#include <ostream>

#define EINSUMS_STRINGIFY(a) EINSUMS_STRINGIFY2(a)
#define EINSUMS_STRINGIFY2(a) #a

#define BEGIN_EINSUMS_NAMESPACE_CPP(x)                                                                                                     \
    namespace x {                                                                                                                          \
    namespace {                                                                                                                            \
    std::string s_Namespace = #x;                                                                                                          \
    }

#define END_EINSUMS_NAMESPACE_CPP(x) }

namespace einsums {

#if defined(MKL_ILP64)
using eint = long long int;
using euint = unsigned long long int;
using elong = long long int;
#else
using eint = int;
using euint = unsigned int;
using elong = long int;
#endif

auto EINSUMS_EXPORT initialize() -> int;
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
    using type = T;
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

template <size_t Rank>
void println(const einsums::Dim<Rank> &dim) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << dim[i] << " ";
    }
    println("Dim{{{}}}", oss.str());
}

template <size_t Rank>
void println(const einsums::Stride<Rank> &stride) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << stride[i] << " ";
    }
    println("Stride{{{}}}", oss.str().c_str());
}

template <size_t Rank>
void println(const einsums::Count<Rank> &count) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << count[i] << " ";
    }
    println("Count{{{}}}", oss.str().c_str());
}

template <size_t Rank>
void println(const einsums::Offset<Rank> &offset) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << offset[i] << " ";
    }
    println("Offset{{{}}}", oss.str().c_str());
}

inline void println(const einsums::Range &range) {
    std::ostringstream oss;
    oss << range[0] << " " << range[1];
    println("Range{{{}}}", oss.str().c_str());
}

template <size_t Rank, typename T>
inline void println(const std::array<T, Rank> &array) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << array[i] << " ";
    }
    println("std::array{{{}}}", oss.str().c_str());
}

// Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define ALIAS_TEMPLATE_FUNCTION(highLevelFunction, lowLevelFunction)                                                                       \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args)->decltype(lowLevelFunction(std::forward<Args>(args)...)) {                               \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }
