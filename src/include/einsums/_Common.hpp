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
 * @file _Common.hpp
 * 
 * Contains definitions that are common across Einsums.
 *
 */

#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

#include "einsums/Error.hpp"
#include "einsums/Print.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_Export.hpp"

// Macro definitions
#define EINSUMS_STRINGIFY(a)  EINSUMS_STRINGIFY2(a)
#define EINSUMS_STRINGIFY2(a) #a

// Begin the namespace and include details.
#define BEGIN_EINSUMS_NAMESPACE_HPP(x)                                         \
    namespace x {                                                              \
    namespace detail {                                                         \
    extern EINSUMS_EXPORT std::string s_Namespace;                             \
    }

// End the namespace.
#define END_EINSUMS_NAMESPACE_HPP(x) }

// Begin the namespace and include details.
#define BEGIN_EINSUMS_NAMESPACE_CPP(x)                                         \
    namespace x {                                                              \
    namespace detail {                                                         \
    EINSUMS_EXPORT std::string s_Namespace = #x;                               \
    }

// End the namespace.
#define END_EINSUMS_NAMESPACE_CPP(x) }

namespace einsums {

// Define types for later use.
#if defined(MKL_ILP64)
using eint  = int128_t;   // long long int;
using euint = uint128_t;  // unsigned long long int;
using elong = int128_t;   // long long int;
#else
using eint  = int;
using euint = unsigned int;
using elong = int64_t;    // long int;
#endif

// Functions to start and end processing of tensor operations.

/**
 * Initialize the environment for running tensor calculations.
 * 
 * @return 0 on success, 1 if there is an MPI error.
 *
 * @callgraph
 * @callergraph
 */
auto EINSUMS_EXPORT initialize() -> int;

/**
 * Finalize the environment for running tensors.
 *
 * @callgraph
 * @callergraph
 */
void EINSUMS_EXPORT finalize(bool timerReport = false);

/* The following detail and "using" statements below are needed to ensure Dims,
 * Strides, and Offsets are strong-types in C++
 */
namespace detail {

struct DimType {};
struct StrideType {};
struct OffsetType {};
struct CountType {};
struct RangeType {};
struct ChunkType {};

/**
 * @class Array
 * @todo Figure this out better.
 * Represents an array of values?
 */
template <typename T, std::size_t Rank, typename UnderlyingType = std::size_t>
struct Array : public std::array<UnderlyingType, Rank> {

    /**
     * Construct an array with entries matching the given arguments.
     * 
     * @callgraph
     * @callergraph
     */
    template <typename... Args>
    constexpr explicit Array(Args... args) :
            std::array<UnderlyingType, Rank>{
                static_cast<UnderlyingType>(args)...} {}
    using type = T;
};
}  // namespace detail

/**
 * @class Dim
 * @todo Find out the use of this class.
 */
template <std::size_t Rank>
using Dim = detail::Array<detail::DimType, Rank, std::int64_t>;

/**
 * @class Stride
 * @todo Find out the use of this class.
 */
template <std::size_t Rank>
using Stride = detail::Array<detail::StrideType, Rank>;

/**
 * @class Offset
 * @todo Find out the use of this class.
 */
template <std::size_t Rank>
using Offset = detail::Array<detail::OffsetType, Rank>;

/**
 * @class Count
 * @todo Find out the use of this class.
 */
template <std::size_t Rank>
using Count = detail::Array<detail::CountType, Rank>;

/**
 * @class Range
 * @todo Find out the use of this class.
 */
using Range = detail::Array<detail::RangeType, 2, std::int64_t>;

/**
 * @class Chunk
 * @todo Find out the use of this class.
 */
template <std::size_t Rank>
using Chunk = detail::Array<detail::ChunkType, Rank, std::int64_t>;

/**
 * @struct All_t
 * @todo Find out the use of this class.
 */
struct All_t {};

/**
 * @var All_t All
 * @todo Find out what this is used for.
 */
static struct All_t All;

}  // namespace einsums

/**
 * Prints the dimensions passed in.
 *
 * @param dim The dimensions to print.
 * 
 * @callgraph
 * @callergraph
 */
template <size_t Rank>
void println(const einsums::Dim<Rank> &dim) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << dim[i] << " ";
    }
    println("Dim{{{}}}", oss.str());
}

/**
 * Prints the strides passed in.
 *
 * @param stride The strides to print.
 * 
 * @callgraph
 * @callergraph
 */
template <size_t Rank>
void println(const einsums::Stride<Rank> &stride) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << stride[i] << " ";
    }
    println("Stride{{{}}}", oss.str().c_str());
}

/**
 * Prints the counts passed in.
 * 
 * @param count The counts to print.
 *
 * @callgraph
 * @callergraph
 */
template <size_t Rank>
void println(const einsums::Count<Rank> &count) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << count[i] << " ";
    }
    println("Count{{{}}}", oss.str().c_str());
}

/**
 * Prints the offsets passed in.
 *
 * @param offset The offset to print.
 * 
 * @callgraph
 * @callergraph
 */
template <size_t Rank>
void println(const einsums::Offset<Rank> &offset) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << offset[i] << " ";
    }
    println("Offset{{{}}}", oss.str().c_str());
}

/**
 * Prints the ranges passed in.
 *
 * @param range The ranges to print.
 *
 * @callgraph
 * @callergraph
 */
inline void println(const einsums::Range &range) {
    std::ostringstream oss;
    oss << range[0] << " " << range[1];
    println("Range{{{}}}", oss.str().c_str());
}

/**
 * Prints the values in an array.
 *
 * @param array The array to print.
 *
 * @callgraph
 * @callergraph
 */
template <size_t Rank, typename T>
inline void println(const std::array<T, Rank> &array) {
    std::ostringstream oss;
    for (size_t i = 0; i < Rank; i++) {
        oss << array[i] << " ";
    }
    println("std::array{{{}}}", oss.str().c_str());
}

/**
 * @def ALIAS_TEMPLATE_FUNCTION(highLevelFunction, lowLevelFunction)
 *
 * Taken from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
 *
 * Creates an alias for the given low-level function as an inline function.
 */
#define ALIAS_TEMPLATE_FUNCTION(highLevelFunction, lowLevelFunction)                                                                       \
    template <typename... Args>                                                                                                            \
    inline auto highLevelFunction(Args &&...args)->decltype(lowLevelFunction(std::forward<Args>(args)...)) {                               \
        return lowLevelFunction(std::forward<Args>(args)...);                                                                              \
    }
