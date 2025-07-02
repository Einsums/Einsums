//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/BLASVendor.hpp>

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

#    include <hip/hip_common.h>
#    include <hip/hip_complex.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {

/**
 * @brief Adjust the index if it is negative and raise an error if it is out of range.
 *
 * The @c index_position parameter is used for creating the exception message. If it is negative,
 * then the exception message will look something like <tt>"The index is out of range! Expected between -5 and 4, got 6!"</tt>.
 * However, for better diagnostics, a non-negative index_position, for example 2, will give an exception message like
 * <tt>"The third index is out of range! Expected between -5 and 4, got 6!"</tt>. Note that these are zero-indexed, so
 * passing in 2 prints out "third".
 *
 * @param index The index to adjust and check.
 * @param dim The dimension to compare to.
 * @param index_position Used for the error message. If it is negative, then the index position
 * will not be included in the error message.
 */
template <std::integral IntType>
constexpr size_t adjust_index(IntType index, size_t dim, int index_position = -1) {
    if constexpr (std::is_signed_v<IntType>) {
        auto hold = index;

        if (hold < 0) {
            hold += dim;
        }

        if (hold < 0 || hold >= dim) {
            if (index_position < 0) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index is out of range! Expected between {} and {}, got {}!",
                                        -(ptrdiff_t)dim, dim - 1, index);
            } else {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index is out of range! Expected between {} and {}, got {}!",
                                        print::ordinal<int>(index_position + 1), -(ptrdiff_t)dim, dim - 1, index);
            }
        }
        return hold;
    } else {
        if (index >= dim) {
            if (index_position < 0) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index is out of range! Expected between {} and {}, got {}!", 0, dim - 1,
                                        index);
            } else {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index is out of range! Expected between {} and {}, got {}!",
                                        print::ordinal<int>(index_position + 1), 0, dim - 1, index);
            }
        }

        return index;
    }
}

template <typename T>
struct TensorImpl;

namespace detail {

/**
 * @struct TensorImpl<T>
 *
 * @brief Underlying implementation details for tensors.
 *
 * @tparam T The data type being stored. It can be const or non-const. It can also be any numerical or complex type, though most
 * library functions only support float, double, std::complex<float>, and std::complex<double>.
 */
template <typename T>
struct TensorImpl final {
  public:
    using ValueType          = T;
    using ReferenceType      = T &;
    using ConstReferenceType = T const &;
    using PointerType        = T *;
    using ConstPointerType   = T const *;

    // Normal constructors. Note that the copy constructor only creates a copy of the view, not a new tensor with the same data.
    constexpr TensorImpl() noexcept = default;

    constexpr TensorImpl(TensorImpl<T> const &other)
        : data_{other.data_}, rank_{other.rank_}, size_{other.size_}, dims_{other.dims_}, strides_{other.strides_},
          is_contiguous_{other.is_contiguous_} {
        gpu_init();
    }

    constexpr TensorImpl(TensorImpl<T> &&other) noexcept
        : data_{other.data_}, rank_{other.rank_}, size_{other.size_}, dims_{std::move(other.dims_)}, strides_{std::move(other.strides_)},
          is_contiguous_{other.is_contiguous_} {
        other.data_ = nullptr;
        other.rank_ = 0;
        other.size_ = 0;
        other.dims_.clear();
        other.strides_.clear();
        gpu_init();
    }

    // Move and copy assignment. Note that the copy assignment only creates a copy of the view, not a new tensor with the same data.
    constexpr TensorImpl<T> &operator=(TensorImpl<T> const &other) {
        data_ = other.data_;
        rank_ = other.rank_;
        size_ = other.size_;
        dims_.resize(rank_);
        strides_.resize(rank_);

        dims_.assign(other.dims_.cbegin(), other.dims_.cend());
        strides_.assign(other.strides_.cbegin(), other.strides_.cend());
        is_contiguous_ = other.is_contiguous_;
        gpu_init();
    }

    constexpr TensorImpl<T> &operator=(TensorImpl<T> &&other) {
        data_          = other.data_;
        rank_          = other.rank_;
        size_          = other.size_;
        dims_          = std::move(other.dims_);
        strides_       = std::move(other.strides_);
        is_contiguous_ = other.is_contiguous_;

        other.data_ = nullptr;
        other.rank_ = 0;
        other.size_ = 0;
        other.dims_.clear();
        other.strides_.clear();
        gpu_init();
    }

    // Destructor.
    constexpr ~TensorImpl() noexcept {
        data_ = nullptr;
        rank_ = 0;
        size_ = 0;
        dims_.clear();
        strides_.clear();
#ifdef EINSUMS_COMPUTE_CODE
        gpu::GPUMemoryTracker::get_singleton().release_handle(gpu_handle_, true);
#endif
    }

    // Now the more useful constructors.
    template <ContainerOrInitializer Dims>
    constexpr TensorImpl(T *data, Dims const &dims, bool row_major = false)
        : data_{data}, dims_(dims.begin(), dims.end()), strides_(dims.size()), rank_{dims.size()} {

        size_          = dims_to_strides(dims_, strides_, row_major);
        is_contiguous_ = true;
    }

    template <ContainerOrInitializer Dims, ContainerOrInitializer Strides>
    constexpr TensorImpl(T *data, Dims const &dims, Strides const &strides)
        : data_{data}, dims_(dims.cbegin(), dims.cend()), strides_(strides.begin(), strides.end()), rank_{dims.size()}, size_{1} {
        for (int i = 0; i < rank_; i++) {
            size_ *= dims_[i];
        }

        // Check to see if it is contiguous.
        if (strides[0] == 1) {
            size_t expected = 1;
            for (int i = 0; i < rank_; i++) {
                if (strides_[i] != expected) {
                    is_contiguous_ = false;
                    break;
                }
                expected *= dims_[i];
            }
            is_contiguous_ = true;
        } else if (strides[rank_ - 1] == 1) {
            size_t expected = 1;
            for (int i = rank_ - 1; i >= 0; i--) {
                if (strides_[i] != expected) {
                    is_contiguous_ = false;
                    break;
                }
                expected *= dims_[i];
            }
            is_contiguous_ = true;
        } else {
            is_contiguous_ = false;
        }
    }

    // Getters

    constexpr size_t rank() const noexcept { return rank_; }

    constexpr size_t size() const noexcept { return size_; }

    constexpr BufferVector<size_t> const &dims() const noexcept { return dims_; }

    constexpr BufferVector<size_t> const &strides() const noexcept { return strides_; }

    constexpr T *data() noexcept { return data_; }

    constexpr T const *data() const noexcept { return data_; }

    constexpr bool is_contiguous() const noexcept { return is_contiguous_; }

    // Setters

    constexpr void reset_data(T *data) noexcept { data_ = data; }

    constexpr void reset_data(T const *data) noexcept { data_ = data; }

    // Indexed getters.

    constexpr size_t dim(int i) const {
        int temp = i;
        if (temp < 0) {
            temp += rank_;
        }

        if (temp < 0 || temp >= rank_) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to dim is out of range! Expected between {} and {}, got {}.",
                                    -rank_, rank_ - 1, i);
        }

        return dims_[temp];
    }

    constexpr size_t stride(int i) const {
        int temp = i;
        if (temp < 0) {
            temp += rank_;
        }

        if (temp < 0 || temp >= rank_) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to stride is out of range! Expected between {} and {}, got {}.",
                                    -rank_, rank_ - 1, i);
        }

        return strides_[temp];
    }

    // Indexed data retrieval.
    template <std::integral... MultiIndex>
    constexpr T *data_no_check(MultiIndex &&...index) {
        return data_no_check(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T *data_no_check(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>([&offset, &index, this](size_t n) { offset += std::get<n>(index) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T *data_no_check(MultiIndex const &index) {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        return data_ + std::inner_product(index.begin(), index.end(), strides_.cbegin(), 0);
    }

    template <std::integral... MultiIndex>
    constexpr T const *data_no_check(MultiIndex &&...index) const {
        return data_no_check(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T const *data_no_check(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>([&offset, &index, this](size_t n) { offset += std::get<n>(index) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T const *data_no_check(MultiIndex const &index) const {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        return data_ + std::inner_product(index.begin(), index.end(), strides_.cbegin(), 0);
    }

    template <std::integral... MultiIndex>
    constexpr T *data(MultiIndex &&...index) {
        return data(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T *data(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>(
            [&offset, &index, this](size_t n) { offset += adjust_index(std::get<n>(index), dims_[n], n) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T *data(MultiIndex const &index) {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        size_t offset = 0;

        for (int i = 0; i < index.size(); i++) {
            offset += adjust_index(index[i], dims_[i], i) * strides_[i];
        }

        return data_ + offset;
    }

    template <std::integral... MultiIndex>
    constexpr T const *data(MultiIndex &&...index) const {
        return data(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T const *data(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>(
            [&offset, &index, this](size_t n) { offset += adjust_index(std::get<n>(index), dims_[n], n) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T const *data(MultiIndex const &index) const {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        size_t offset = 0;

        for (int i = 0; i < index.size(); i++) {
            offset += adjust_index(index[i], dims_[i], i) * strides_[i];
        }

        return data_ + offset;
    }

    // Const conversion.
    constexpr operator TensorImpl<T const>() { return TensorImpl<T const>(data_, dims_, strides_); }

    // Subscripting.
    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript_no_check(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript_no_check(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ReferenceType subscript_no_check(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript_no_check(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript_no_check(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ConstReferenceType subscript_no_check(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ReferenceType subscript(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ConstReferenceType subscript(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    // View creation.
    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript_no_check(MultiIndex &&...index) {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript_no_check(std::tuple<MultiIndex...> const &index) {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, index);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript_no_check(MultiIndex &&...index) const {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript_no_check(std::tuple<MultiIndex...> const &index) const {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, index);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript(MultiIndex &&...index) {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript(std::tuple<MultiIndex...> const &index) {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, index);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript(MultiIndex &&...index) const {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript(std::tuple<MultiIndex...> const &index) const {
        BufferVector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, index);

        return TensorImpl<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename TOther>
    void copy_from_both_contiguous(TensorImpl<TOther> const &other) {
        if (other.rank() != rank()) {
            EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
        }

        if (other.dims() != dims()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
        }

        if (other.data() == data() || data() == nullptr) {
            // Don't copy.
            return;
        }

        size_t elems = size();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < elems; i++) {
            if constexpr (!IsComplexV<T>) {
                data_[i] = T{std::real(other.data_[i])};
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
                auto val = other.data_[i];
                data_[i] = T{std::real(val), std::imag(val)};
            } else {
                data_[i] = T{other.data_[i]};
            }
        }
    }

    template <typename TOther>
    void copy_from(TensorImpl<TOther> const &other) {
        if (other.is_contiguous() && is_contiguous()) {
            copy_from_assume_contiguous(other);
            return;
        }

        if (other.rank() != rank()) {
            EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
        }

        if (other.dims() != dims()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
        }

        if (other.data() == data() || data() == nullptr) {
            // Don't copy.
            return;
        }

        size_t elems = size();

        BufferVector<size_t> index_strides;

        dims_to_strides(dims(), index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < elems; i++) {
            size_t this_sentinel, other_sentinel;

            sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

            if constexpr (!IsComplexV<T>) {
                data_[this_sentinel] = T{std::real(other.data_[other_sentinel])};
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
                auto val             = other.data_[other_sentinel];
                data_[this_sentinel] = T{std::real(val), std::imag(val)};
            } else {
                data_[this_sentinel] = T{other.data_[other_sentinel]};
            }
        }
    }

    template <typename TOther>
    void add_assign_both_contiguous(TensorImpl<TOther> const &other) {
        if (other.rank() != rank()) {
            EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
        }

        if (other.dims() != dims()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
        }

        if (other.data() == data() || data() == nullptr) {
            // Don't copy.
            return;
        }

        if constexpr (std::is_same_v<TOther, T>) {
            #ifdef EINSUMS_COMPUTE_CODE
            auto singleton = gpu::GPUMemoryTracker::get_singleton();
            if(singleton.handle_is_allocated(gpu_handle_) && singleton.handle_is_allocated(other.gpu_handle_)) {
                hipblas_catch(hipblasSaxpy());
            }
            #endif
            blas::vendor::axpy(size(), );
        }

        size_t elems = size();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < elems; i++) {
            if constexpr (!IsComplexV<T>) {
                data_[i] = T{std::real(other.data_[i])};
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
                auto val = other.data_[i];
                data_[i] = T{std::real(val), std::imag(val)};
            } else {
                data_[i] = T{other.data_[i]};
            }
        }
    }

    template <typename TOther>
    void add_assign(TensorImpl<TOther> const &other) {
        if (other.is_contiguous() && is_contiguous()) {
            copy_from_assume_contiguous(other);
            return;
        }

        if (other.rank() != rank()) {
            EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
        }

        if (other.dims() != dims()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
        }

        if (other.data() == data() || data() == nullptr) {
            // Don't copy.
            return;
        }

        size_t elems = size();

        BufferVector<size_t> index_strides;

        dims_to_strides(dims(), index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < elems; i++) {
            size_t this_sentinel, other_sentinel;

            sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

            if constexpr (!IsComplexV<T>) {
                data_[this_sentinel] = T{std::real(other.data_[other_sentinel])};
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
                auto val             = other.data_[other_sentinel];
                data_[this_sentinel] = T{std::real(val), std::imag(val)};
            } else {
                data_[this_sentinel] = T{other.data_[other_sentinel]};
            }
        }
    }

#ifdef EINSUMS_COMPUTE_CODE

    void copy_to_gpu(gpu::GPUPointer<T> gpu_ptr) {
        BufferVector<size_t> index_strides(rank_);

        dims_to_strides(dims_, index_strides);

        if (size_ < 2048) {
            BufferVector<T> buffer(size_);

            for (size_t i = 0; i < size_; i++) {
                size_t sentinel;

                sentinel_to_sentinels(i, index_strides, strides_, sentinel);

                buffer[i] = data_[sentinel];
            }

            hip_catch(hipMemcpy((void *)gpu_ptr, (void const *)buffer.data(), size_ * sizeof(T), hipMemcpyHostToDevice));
        } else {
            // Double buffer bigger transactions.
            BufferVector<T>       buffer1(1024), buffer2(1024);
            std::binary_semaphore buffer1_semaphore(1), buffer2_semaphore(1);
            bool                  buffer1_ready = false, buffer2_ready = false;

#    pragma omp parallel
            {
#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();
                        for (size_t j = 0; j < std::min(size_ - i, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j, index_strides, strides_, sentinel);
                            buffer1[j] = data_[sentinel];
                        }

                        if (size_ - i < 1024) {
                            buffer1.resize(size_ - i);
                        }

                        buffer1_ready = true;

                        buffer1_semaphore.release();

                        while (buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        for (size_t j = 0; j < std::min(size_ - i - 1024, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j + 1024, index_strides, strides_, sentinel);
                            buffer2[j] = data_[sentinel];
                        }

                        if (size_ - i - 1024 < 1024) {
                            buffer2.resize(size_ - i);
                        }

                        buffer2_ready = true;

                        buffer2_semaphore.release();
                    }
                }

#    pragma omp task
                {
                    std::this_thread::yield();
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (!buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();

                        hip_catch(
                            hipMemcpy((void *)(gpu_ptr + i), (void *)buffer1.data(), buffer1.size() * sizeof(T), hipMemcpyHostToDevice));

                        buffer1_ready = false;

                        buffer1_semaphore.release();

                        while (!buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)(gpu_ptr + i + 1024), (void *)buffer2.data(), buffer2.size() * sizeof(T),
                                            hipMemcpyHostToDevice));

                        buffer2_ready = false;

                        buffer2_semaphore.release();
                    }
                }
#    pragma omp taskgroup
            }
        }
    }

    void copy_from_gpu(gpu::GPUPointer<T> gpu_ptr) {

        BufferVector<size_t> index_strides(rank_);

        dims_to_strides(dims_, index_strides);

        if (size_ < 2048) {
            BufferVector<T> buffer(size_);

            hip_catch(hipMemcpy((void *)buffer.data(), (void const *)gpu_ptr, size_ * sizeof(T), hipMemcpyDeviceToHost));

            for (size_t i = 0; i < size_; i++) {
                size_t sentinel;

                sentinel_to_sentinels(i, index_strides, strides_, sentinel);

                buffer[i]       = data_[sentinel];
                data_[sentinel] = buffer[i];
            }

        } else {
            // Double buffer bigger transactions.
            BufferVector<T>       buffer1(1024), buffer2(1024);
            std::binary_semaphore buffer1_semaphore(1), buffer2_semaphore(1);
            bool                  buffer1_ready = true, buffer2_ready = true;

#    pragma omp parallel
            {
#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();
                        for (size_t j = 0; j < std::min(size_ - i, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j, index_strides, strides_, sentinel);
                            data_[sentinel] = buffer1[j];
                        }

                        buffer1_ready = true;

                        buffer1_semaphore.release();

                        while (buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        for (size_t j = 0; j < std::min(size_ - i - 1024, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j + 1024, index_strides, strides_, sentinel);
                            buffer2[j]      = data_[sentinel];
                            data_[sentinel] = buffer2[j];
                        }

                        buffer2_ready = true;

                        buffer2_semaphore.release();
                    }
                }

#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (!buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)buffer1.data(), (void const *)(gpu_ptr + i), buffer1.size() * sizeof(T),
                                            hipMemcpyDeviceToHost));

                        buffer1_ready = false;

                        buffer1_semaphore.release();

                        while (!buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)buffer2.data(), (void const *)(gpu_ptr + i + 1024), buffer2.size() * sizeof(T),
                                            hipMemcpyDeviceToHost));

                        buffer2_ready = false;

                        buffer2_semaphore.release();
                    }
                }
#    pragma omp taskgroup
            }
        }
    }

    gpu::GPUPointer<T> request_gpu_ptr() {
        auto [ptr, do_copy] = gpu::GPUMemoryTracker::get_singleton().get_pointer<T>(gpu_handle_, size());

        if (do_copy) {
            copy_to_gpu(ptr);
        }

        return ptr;
    }

    void release_gpu_ptr(gpu::GPUPointer<T> &ptr) {
        if (ptr) {
            gpu::GPUMemoryTracker::get_singleton().release_handle(gpu_handle_);
        }
    }
#endif

  private:
    void gpu_init() {
#ifdef EINSUMS_COMPUTE_CODE
        gpu_handle_ = gpu::GPUMemoryTracker::get_singleton().create_handle();
#endif
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides) {
        return 0;
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, std::integral First, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides, First first,
                                  Rest &&...rest) {
        auto index = first;
        if constexpr (CheckInds) {
            index = adjust_index(index, dims_[I], I);
        }

        return index * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides, Range const &first,
                                  Rest &&...rest) {
        auto index = first;
        if constexpr (CheckInds) {
            index[0] = adjust_index(index[0], dims_[I], I);
            index[1] = adjust_index(index[1], dims_[I] + 1, I);
        }

        if (index[0] > index[1]) {
            auto temp = index[0];
            index[0]  = index[1];
            index[1]  = temp;
        }

        out_dims.push_back(index[1] - index[0]);
        out_strides.push_back(strides_[I]);

        return index[0] * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides, AllT const &,
                                  Rest &&...rest) {

        out_dims.push_back(dims_[I]);
        out_strides.push_back(strides_[I]);

        return compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, typename... MultiIndex>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides,
                                  std::tuple<MultiIndex...> const &indices) {
        using CurrType = typename std::tuple_element_t<I, std::tuple<MultiIndex...>>;
        size_t index   = 0;
        if constexpr (I >= sizeof...(MultiIndex)) {
            return 0;
        } else {
            if constexpr (std::is_integral_v<CurrType>) {
                index = std::get<I>(indices);
                if constexpr (CheckInds) {
                    index = adjust_index(index, dims_[I], I);
                }

            } else if constexpr (std::is_same_v<Range, CurrType>) {
                auto range = std::get<I>(indices);
                if constexpr (CheckInds) {
                    range[0] = adjust_index(range[0], dims_[I], I);
                    range[1] = adjust_index(range[1], dims_[I] + 1, I);
                }

                if (range[0] > range[1]) {
                    auto temp = range[0];
                    range[0]  = range[1];
                    range[1]  = temp;
                }

                index = range[0];
                out_dims.push_back(range[1] - range[0]);
                out_strides.push_back(strides_[I]);
            } else if constexpr (std::is_same_v<AllT, CurrType>) {
                out_dims.push_back(dims_[I]);
                out_strides.push_back(strides_[I]);
            }
            return index * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, indices);
        }
    }

    T *data_{nullptr};

    size_t rank_{0}, size_{0};

    BufferVector<size_t> dims_{}, strides_{};

#ifdef EINSUMS_COMPUTE_CODE
    size_t gpu_handle_{0};
#endif

    bool is_contiguous_{false};
};

} // namespace detail

} // namespace einsums