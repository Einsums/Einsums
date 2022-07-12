#pragma once

#include <array>
#include <cstdint>

namespace einsums {

// The following detail and "using" statements below are needed to ensure Dims, Strides, and Offsets are strong-types in C++
namespace detail {

struct DimType {};
struct StrideType {};
struct OffsetType {};
struct CountType {};
struct RangeType {};

template <typename T, std::size_t Rank, typename UnderlyingType = std::size_t>
struct Array : public std::array<UnderlyingType, Rank> {
    template <typename... Args>
    constexpr explicit Array(Args... args) : std::array<UnderlyingType, Rank>{static_cast<UnderlyingType>(args)...} {}
    using type = T;
};
} // namespace detail

template <std::size_t Rank>
using Dim = detail::Array<detail::DimType, Rank>;

template <std::size_t Rank>
using Stride = detail::Array<detail::StrideType, Rank>;

template <std::size_t Rank>
using Offset = detail::Array<detail::OffsetType, Rank>;

template <std::size_t Rank>
using Count = detail::Array<detail::CountType, Rank>;

using Range = detail::Array<detail::RangeType, 2, std::int64_t>;

struct All_t {};
static struct All_t All;

} // namespace einsums