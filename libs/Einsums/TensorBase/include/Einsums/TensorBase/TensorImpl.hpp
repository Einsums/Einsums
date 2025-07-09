//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/TensorBase/Common.hpp>

#include <type_traits>

namespace einsums {
namespace detail {

template <typename TemplateType, typename OutputType>
struct MakePointerLike {
    using type = OutputType *;
};

template <typename TemplateType, typename OutputType>
struct MakePointerLike<TemplateType const, OutputType> {
    using type = std::add_const_t<OutputType> *;
};

template <typename TemplateType, typename OutputType>
struct MakePointerLike<TemplateType volatile, OutputType> {
    using type = std::add_volatile_t<OutputType> *;
};

template <typename TemplateType, typename OutputType>
struct MakePointerLike<TemplateType const volatile, OutputType> {
    using type = std::add_cv_t<OutputType> *;
};

template <typename T>
struct TensorImpl final {
  public:
    using pointer            = T *;
    using const_pointer      = std::add_const_t<T> *;
    using void_pointer       = typename MakePointerLike<T, void>::type;
    using const_void_pointer = typename MakePointerLike<std::add_const_t<T>, void>::type;
    using value_type         = T;
    using reference          = T &;
    using const_reference    = std::add_const_t<T> &;

    // Rule of five methods.

    constexpr TensorImpl() noexcept : ptr_{nullptr}, dims_(), strides_(), rank_{0}, size_{0} {};

    template <typename TOther>
        requires requires {
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<T>; // We don't want to make a non-const copy of a const impl.
        }
    constexpr TensorImpl(TensorImpl<TOther> const &other)
        : ptr_{other.ptr_}, rank_{other.rank_}, strides_{other.strides_}, dims_{other.dims_}, size_{other.size_} {}

    template <typename TOther>
        requires requires {
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<T>; // We don't want to make a non-const copy of a const impl.
        }
    constexpr TensorImpl(TensorImpl<TOther> &&other) noexcept
        : ptr_{other.ptr_}, rank_{other.rank_}, strides_{std::move(other.strides_)}, dims_{std::move(other.dims_)}, size_{other.size_} {
        other.ptr_  = nullptr;
        other.rank_ = 0;
        other.strides_.clear();
        other.dims_.clear();
        other.size_ = 0;
    }

    template <typename TOther>
        requires requires {
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<T>; // We don't want to make a non-const copy of a const impl.
        }
    constexpr TensorImpl<T> &operator=(TensorImpl<TOther> const &other) {
        ptr_     = other.ptr_;
        rank_    = other.rank_;
        strides_ = other.strides_;
        dims_    = other.dims_;
        size_    = other.size_;
        return *this;
    }

    template <typename TOther>
        requires requires {
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<T>; // We don't want to make a non-const copy of a const impl.
        }
    constexpr TensorImpl<T> &operator=(TensorImpl<TOther> &&other) noexcept {
        ptr_     = other.ptr_;
        rank_    = other.rank_;
        strides_ = std::move(other.strides_);
        dims_    = std::move(other.dims_);
        size_    = other.size_;

        other.ptr_  = nullptr;
        other.rank_ = 0;
        other.strides_.clear();
        other.dims_.clear();
        other.size_ = 0;
        return *this;
    }

    // Other constructors.
    template <ContainerOrInitializer Dims>
    constexpr TensorImpl(pointer ptr, Dims const &dims, bool row_major = false)
        : ptr_{ptr}, rank_{dims.size()}, dims_(dims.begin(), dims.end()) {
        strides_.resize(rank_);

        size_ = 1;

        if (row_major) {
            for (int i = rank_ - 1; i >= 0; i--) {
                strides_[i] = size_;
                size_ *= dims_[i];
            }
        } else {
            for (int i = 0; i < rank_; i++) {
                strides_[i] = size_;
                size_ *= dims_[i];
            }
        }
    }

    template <ContainerOrInitializer Dims, ContainerOrInitializer Strides>
    constexpr TensorImpl(pointer ptr, Dims const &dims, Strides const &strides)
        : ptr_{ptr}, rank_{dims.size()}, dims_(dims.begin(), dims.end()), strides_(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>())} {}

    // Getters and setters.

    constexpr pointer data() noexcept { return ptr_; }

    constexpr const_pointer data() const noexcept { return ptr_; }

    constexpr size_t rank() const noexcept { return rank_; }

    constexpr BufferVector<size_t> dims() const noexcept { return dims_; }

    constexpr BufferVector<size_t> strides() const noexcept { return strides_; }

    constexpr size_t size() const noexcept { return size_; }

    constexpr void set_data(pointer *ptr) noexcept { ptr_ = ptr; }

    // Indexed getters and setters.

    template <std::integral... MultiIndex>
    constexpr pointer data(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel_negative_check(strides_, dims_, {index...});
    }

    template <std::integral... MultiIndex>
    constexpr const_pointer data(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel_negative_check(strides_, dims_, {index...});
    }

    template <std::integral... MultiIndex>
    constexpr pointer data_no_check(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, {index...});
    }

    template <std::integral... MultiIndex>
    constexpr const_pointer data_no_check(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_seintinel(strides_, {index...});
    }

    template <ContainerOrInitializer Index>
    constexpr pointer data(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel_negative_check(strides_, dims_, index);
    }

    template <ContainerOrInitializer Index>
    constexpr const_pointer data(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel_negative_check(strides_, dims_, index);
    }

    template <ContainerOrInitializer Index>
    constexpr pointer data_no_check(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, dims_, index);
    }

    template <ContainerOrInitializer Index>
    constexpr const_pointer data_no_check(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, dims_, index);
    }

    constexpr size_t dim(std::integral auto i) const {
        if (rank_ == 0) {
            return 1;
        }
        auto temp = i;
        if (temp < 0) {
            temp += rank_;
        }

        if (temp < 0 || temp >= rank_) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to dim was out of range! Got {}, expected between {} and {}.", i,
                                    -(ptrdiff_t)rank_, rank_ - 1);
        }

        return dims_[i];
    }

    constexpr size_t stride(std::integral auto i) const {
        if (rank_ == 0) {
            return 0;
        }
        auto temp = i;
        if (temp < 0) {
            temp += rank_;
        }

        if (temp < 0 || temp >= rank_) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to stride was out of range! Got {}, expected between {} and {}.",
                                    i, -(ptrdiff_t)rank_, rank_ - 1);
        }

        return strides_[i];
    }

    // More complicated getters.

    constexpr bool is_contiguous() const {
        if (rank_ == 0) {
            return true;
        }
        if (stride(0) < stride(-1)) {
            return dim(-1) * stride(-1) == size_;
        } else {
            return dim(0) * stride(0) == size_;
        }
    }

    bool is_totally_vectorable(size_t *incx = nullptr) const {
        if (rank_ == 0) {
            if (incx != nullptr) {
                *incx = 0;
            }
            return false;
        } else if (rank_ == 1) {
            if (incx != nullptr) {
                *incx = stride(0);
            }
            return true;
        } else {
            if (stride(0) < stride(-1)) {
                if (incx != nullptr) {
                    *incx = stride(0);
                }
                return dim(-1) * stride(-1) / stride(0) == size_;
            } else {
                if (incx != nullptr) {
                    *incx = stride(-1);
                }
                return dim(0) * stride(0) / stride(-1) == size_;
            }
        }
    }

    bool is_gemmable(size_t *lda = nullptr) const {
        if (rank_ != 2) {
            return false;
        } else if (strides_[0] != 1 && strides_[1] != 1) {
            return false;
        } else {
            if (lda != nullptr) {
                *lda = std::max(strides_[0], strides_[1]);
            }
            return true;
        }
    }

    constexpr bool is_row_major() const {
        if (rank_ <= 1) {
            return true;
        } else {
            return stride(-1) < stride(0);
        }
    }

    constexpr bool is_column_major() const {
        if (rank_ <= 1) {
            return true;
        } else {
            return stride(-1) > stride(0);
        }
    }

    template <Container IndexStrides>
    void query_vectorable_params(size_t *easy_size, size_t *hard_size, size_t *easy_rank, size_t *incx,
                                 IndexStrides &all_index_strides) const {
        all_index_strides.resize(rank_);
        if (rank_ == 0) {
            *easy_size = 0;
            *hard_size = 0;
            *easy_rank = 0;
            *incx      = 0;
        } else if (rank_ == 1) {
            *easy_size           = size_;
            *hard_size           = 0;
            *easy_rank           = 1;
            *incx                = strides_[0];
            all_index_strides[0] = 1;
        } else {

            *easy_size = 1;
            *hard_size = 1;
            *easy_rank = 0;

            if (is_row_major()) {
                *incx       = stride(-1);
                size_t size = 1;
                for (int i = rank_ - 1; i >= 0; i--) {
                    all_index_strides[i] = size;

                    if (size * *incx == strides_[i]) {
                        *easy_rank += 1;
                        *easy_size *= dims_[i];
                    } else {
                        *hard_size *= dims_[i];
                    }
                }
            } else {
                *incx       = stride(0);
                size_t size = 1;
                for (int i = 0; i < rank_; i++) {
                    all_index_strides[i] = size;

                    if (size * *incx == strides_[i]) {
                        *easy_rank += 1;
                        *easy_size *= dims_[i];
                    } else {
                        *hard_size *= dims_[i];
                    }
                }
            }
        }
    }

    // Subscript.

    template <std::integral... MultiIndex>
    constexpr reference subscript(MultiIndex &&...index) {
        return *data(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr const_reference subscript(MultiIndex &&...index) const {
        return *data(std::forward<MultiIndex>(index)...);
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr reference subscript(MultiIndex const &index) {
        return *data(index);
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr const_reference subscript(MultiIndex const &index) const {
        return *data(index);
    }

    template <std::integral... MultiIndex>
    constexpr reference subscript_no_check(MultiIndex &&...index) {
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr const_reference subscript_no_check(MultiIndex &&...index) const {
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr reference subscript_no_check(MultiIndex const &index) {
        return *data_no_check(index);
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr const_reference subscript_no_check(MultiIndex const &index) const {
        return *data_no_check(index);
    }

    // Slicing.

    template <typename... MultiIndex>
        requires((std::is_integral_v<MultiIndex> || std::is_base_of_v<Range, MultiIndex> || std::is_base_of_v<AllT, MultiIndex>) && ... &&
                 true)
    constexpr TensorImpl<T> subscript(MultiIndex &&...index) {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

        adjust_ranges<0>(index_tuple);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view<0>(new_dims, new_strides, index_tuple);

        if (ptr_ == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(ptr_ + offset, new_dims, new_strides);
        }
    }

    template <typename... MultiIndex>
        requires((std::is_integral_v<MultiIndex> || std::is_base_of_v<Range, MultiIndex> || std::is_base_of_v<AllT, MultiIndex>) && ... &&
                 true)
    constexpr TensorImpl<std::add_const_t<T>> subscript(MultiIndex &&...index) const {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

        adjust_ranges<0>(index_tuple);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view<0>(new_dims, new_strides, index_tuple);

        if (ptr_ == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(ptr_ + offset, new_dims, new_strides);
        }
    }

  private:
    template <size_t I, typename... MultiIndex>
    constexpr void adjust_ranges(std::tuple<MultiIndex...> &indices) const {
        if constexpr (I >= sizeof...(MultiIndex)) {
            return;
        } else if constexpr (std::is_integral_v<std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
            auto &index = std::get<I>(indices);
            auto  temp  = index;

            if (index < 0) {
                index += dims_[I];
            }

            if (index < 0 || index >= dims_[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Index passed to view creation is out of range! Got {}, expected between {} and {}.", temp,
                                        -(ptrdiff_t)dims_[I], dims_[I] - 1);
            }
            adjust_ranges<I + 1>(indices);
        } else if constexpr (std::is_base_of_v<Range, std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
            auto &index = std::get<I>(indices);
            auto  temp  = index;

            if (index[0] < 0) {
                index[0] += dims_[I];
            }

            if (index[1] < 0) {
                index[1] += dims_[I];
            }

            if (index[0] < 0 || index[0] >= dims_[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Lower bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[0], -(ptrdiff_t)dims_[I], dims_[I] - 1);
            }

            if (index[1] < 0 || index[1] > dims_[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Upper bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[1], -(ptrdiff_t)dims_[I], dims_[I]);
            }
            adjust_ranges<I + 1>(indices);
        } else {
            adjust_ranges<I + 1>(indices);
        }
    }

    template <size_t I, typename... MultiIndex>
        requires((std::is_integral_v<MultiIndex> || std::is_base_of_v<Range, MultiIndex> || std::is_base_of_v<AllT, MultiIndex>) && ... &&
                 true)
    constexpr size_t compute_view(BufferVector<size_t> &out_dims, BufferVector<size_t> &out_strides,
                                  std::tuple<MultiIndex...> const &indices) const {
        if constexpr (I >= sizeof...(MultiIndex)) {
            return 0;
        } else {
            if constexpr (std::is_integral_v<std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
                return std::get<I>(indices) * strides_[I] + compute_view<I + 1>(out_dims, out_strides, indices);
            } else if constexpr (std::is_base_of_v<Range, std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
                Range range = std::get<I>(indices);

                out_dims.push_back(range[1] - range[0]);
                out_strides.push_back(strides_[I]);
                return range[0] * strides_[I] + compute_view<I + 1>(out_dims, out_strides, indices);
            } else if constexpr (std::is_base_of_v<AllT, std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
                out_dims.push_back(dims_[I]);
                out_strides.push_back(strides_[I]);
                return compute_view<I + 1>(out_dims, out_strides, indices);
            }
        }
    }

    pointer              ptr_{nullptr};
    size_t               rank_, size_;
    BufferVector<size_t> dims_, strides_;
};

} // namespace detail
} // namespace einsums