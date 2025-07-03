//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/Backends/TensorImpl.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUMemory/GPUAllocator.hpp>
#    include <Einsums/GPUMemory/GPUMemoryTracker.hpp>
#    include <Einsums/GPUMemory/GPUPointer.hpp>
#    include <Einsums/GPUStreams/GPUStreams.hpp>
#    include <Einsums/hipBLAS.hpp>

#    include <hip/hip_common.h>
#    include <hip/hip_complex.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {
namespace detail {

// Indexed data retrieval.

template <typename T>
template <std::integral... MultiIndex>
constexpr T *TensorImpl<T>::data_no_check(MultiIndex &&...index) {
    return data_no_check(std::make_tuple(std::forward<MultiIndex>(index)...));
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T *TensorImpl<T>::data_no_check(std::tuple<MultiIndex...> const &index) {
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
    }

    size_t offset = 0;
    for_sequence<sizeof...(MultiIndex)>([&offset, &index, this](size_t n) { offset += std::get<n>(index) * strides_[n]; });

    return data_ + offset;
}

template <typename T>
template <ContainerOrInitializer MultiIndex>
constexpr T *TensorImpl<T>::data_no_check(MultiIndex const &index) {
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
    }

    return data_ + std::inner_product(index.begin(), index.end(), strides_.cbegin(), 0);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T const *TensorImpl<T>::data_no_check(MultiIndex &&...index) const {
    return data_no_check(std::make_tuple(std::forward<MultiIndex>(index)...));
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T const *TensorImpl<T>::data_no_check(std::tuple<MultiIndex...> const &index) const {
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
    }

    size_t offset = 0;
    for_sequence<sizeof...(MultiIndex)>([&offset, &index, this](size_t n) { offset += std::get<n>(index) * strides_[n]; });

    return data_ + offset;
}

template <typename T>
template <ContainerOrInitializer MultiIndex>
constexpr T const *TensorImpl<T>::data_no_check(MultiIndex const &index) const {
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
    }

    return data_ + std::inner_product(index.begin(), index.end(), strides_.cbegin(), 0);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T *TensorImpl<T>::data(MultiIndex &&...index) {
    return data(std::make_tuple(std::forward<MultiIndex>(index)...));
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T *TensorImpl<T>::data(std::tuple<MultiIndex...> const &index) {
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
    }

    size_t offset = 0;
    for_sequence<sizeof...(MultiIndex)>(
        [&offset, &index, this](size_t n) { offset += adjust_index(std::get<n>(index), dims_[n], n) * strides_[n]; });

    return data_ + offset;
}

template <typename T>
template <ContainerOrInitializer MultiIndex>
constexpr T *TensorImpl<T>::data(MultiIndex const &index) {
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
    }

    size_t offset = 0;

    for (int i = 0; i < index.size(); i++) {
        offset += adjust_index(index[i], dims_[i], i) * strides_[i];
    }

    return data_ + offset;
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T const *TensorImpl<T>::data(MultiIndex &&...index) const {
    return data(std::make_tuple(std::forward<MultiIndex>(index)...));
}

template <typename T>
template <std::integral... MultiIndex>
constexpr T const *TensorImpl<T>::data(std::tuple<MultiIndex...> const &index) const {
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
    }

    size_t offset = 0;
    for_sequence<sizeof...(MultiIndex)>(
        [&offset, &index, this](size_t n) { offset += adjust_index(std::get<n>(index), dims_[n], n) * strides_[n]; });

    return data_ + offset;
}

template <typename T>
template <ContainerOrInitializer MultiIndex>
constexpr T const *TensorImpl<T>::data(MultiIndex const &index) const {
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
    }

    size_t offset = 0;

    for (int i = 0; i < index.size(); i++) {
        offset += adjust_index(index[i], dims_[i], i) * strides_[i];
    }

    return data_ + offset;
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ReferenceType TensorImpl<T>::subscript_no_check(MultiIndex &&...index) {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data_no_check(std::forward<MultiIndex>(index)...);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ReferenceType TensorImpl<T>::subscript_no_check(std::tuple<MultiIndex...> const &index) {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data_no_check(index);
}

template <typename T>
template <ContainerOrInitializer Index>
constexpr TensorImpl<T>::ReferenceType TensorImpl<T>::subscript_no_check(Index const &index) {
    if (index.size() < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data_no_check(index);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ConstReferenceType TensorImpl<T>::subscript_no_check(MultiIndex &&...index) const {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data_no_check(std::forward<MultiIndex>(index)...);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ConstReferenceType TensorImpl<T>::subscript_no_check(std::tuple<MultiIndex...> const &index) const {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data_no_check(index);
}

template <typename T>
template <ContainerOrInitializer Index>
constexpr TensorImpl<T>::ConstReferenceType TensorImpl<T>::subscript_no_check(Index const &index) const {
    if (index.size() < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data_no_check(index);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ReferenceType TensorImpl<T>::subscript(MultiIndex &&...index) {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data(std::forward<MultiIndex>(index)...);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ReferenceType TensorImpl<T>::subscript(std::tuple<MultiIndex...> const &index) {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data(index);
}

template <typename T>
template <ContainerOrInitializer Index>
constexpr TensorImpl<T>::ReferenceType TensorImpl<T>::subscript(Index const &index) {
    if (index.size() < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data(index);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ConstReferenceType TensorImpl<T>::subscript(MultiIndex &&...index) const {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data(std::forward<MultiIndex>(index)...);
}

template <typename T>
template <std::integral... MultiIndex>
constexpr TensorImpl<T>::ConstReferenceType TensorImpl<T>::subscript(std::tuple<MultiIndex...> const &index) const {
    if (sizeof...(MultiIndex) < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (sizeof...(MultiIndex) > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data(index);
}

template <typename T>
template <ContainerOrInitializer Index>
constexpr TensorImpl<T>::ConstReferenceType TensorImpl<T>::subscript(Index const &index) const {
    if (index.size() < rank_) {
        EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
    }
    if (index.size() > rank_) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
    }
    return *data(index);
}

// View creation.
template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> TensorImpl<T>::subscript_no_check(MultiIndex &&...index) {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<false, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> TensorImpl<T>::subscript_no_check(std::tuple<MultiIndex...> const &index) {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<false, 0>(out_dims, out_strides, index);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> const TensorImpl<T>::subscript_no_check(MultiIndex &&...index) const {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<false, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> const TensorImpl<T>::subscript_no_check(std::tuple<MultiIndex...> const &index) const {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<false, 0>(out_dims, out_strides, index);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> TensorImpl<T>::subscript(MultiIndex &&...index) {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<true, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> TensorImpl<T>::subscript(std::tuple<MultiIndex...> const &index) {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<true, 0>(out_dims, out_strides, index);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> const TensorImpl<T>::subscript(MultiIndex &&...index) const {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<true, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}

template <typename T>
template <typename... MultiIndex>
    requires(!std::is_integral_v<MultiIndex> || ... || false)
constexpr TensorImpl<T> const TensorImpl<T>::subscript(std::tuple<MultiIndex...> const &index) const {
    BufferVector<size_t> out_dims{}, out_strides{};

    out_dims.reserve(rank_);
    out_strides.reserve(rank_);

    size_t offset = compute_view<true, 0>(out_dims, out_strides, index);

    return TensorImpl<T>(data_ + offset, out_dims, out_strides);
}
} // namespace detail
} // namespace einsums