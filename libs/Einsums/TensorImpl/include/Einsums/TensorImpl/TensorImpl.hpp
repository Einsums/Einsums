//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

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

/**
 * @struct TensorImpl
 *
 * @brief Underlying implementation details for tensor objects.
 */
template <typename T>
struct TensorImpl final {
  public:
    /**
     * @typedef pointer
     *
     * @brief The type for pointers returned by this class.
     */
    using pointer = T *;

    /**
     * @typedef const_pointer
     *
     * @brief The type for const pointers returned by this class.
     */
    using const_pointer = std::add_const_t<T> *;

    /**
     * @typedef void_pointer
     *
     * @brief The type for void pointers returned by this class.
     */
    using void_pointer = typename MakePointerLike<T, void>::type;

    /**
     * @typedef const_void_pointer
     *
     * @brief The type for const void pointers returned by this class.
     */
    using const_void_pointer = typename MakePointerLike<std::add_const_t<T>, void>::type;

    /**
     * @typedef value_type
     *
     * @brief The type of data stored by this class.
     */
    using value_type = T;

    /**
     * @typedef reference
     *
     * @brief The reference type returned by this class.
     */
    using reference = T &;

    /**
     * @typedef const_reference
     *
     * @brief The const reference type returned by this class.
     */
    using const_reference = std::add_const_t<T> &;

    // Rule of five methods.

    /**
     * @brief Default constructor.
     */
    constexpr TensorImpl() noexcept : ptr_{nullptr}, dims_(), strides_(), rank_{0}, size_{0} {};

    /**
     * @brief Copy constructor.
     */
    constexpr TensorImpl(TensorImpl<T> const &other)
        : ptr_{other.ptr_}, rank_{other.rank_}, strides_{other.strides_}, dims_{other.dims_}, size_{other.size_} {}

    /**
     * @brief Move constructor.
     */
    constexpr TensorImpl(TensorImpl<T> &&other) noexcept
        : ptr_{other.ptr_}, rank_{other.rank_}, strides_{std::move(other.strides_)}, dims_{std::move(other.dims_)}, size_{other.size_} {
        other.ptr_  = nullptr;
        other.rank_ = 0;
        other.strides_.clear();
        other.dims_.clear();
        other.size_ = 0;
    }

    /**
     * @brief Copy assignment.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> const &other) {
        ptr_     = other.ptr_;
        rank_    = other.rank_;
        strides_ = other.strides_;
        dims_    = other.dims_;
        size_    = other.size_;
        return *this;
    }

    /**
     * @brief Move assignment.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> &&other) noexcept {
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

    /**
     * @brief Copy constructor from a datatype with different constness.
     */
    template <typename TOther>
        requires requires {
            requires !std::is_same_v<T, TOther>;
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<TOther>;
        }
    constexpr TensorImpl(TensorImpl<TOther> const &other)
        : ptr_{other.ptr_}, rank_{other.rank_}, strides_{other.strides_}, dims_{other.dims_}, size_{other.size_} {}

    /**
     * @brief Move constructor from a datatype with different constness.
     */
    template <typename TOther>
        requires requires {
            requires !std::is_same_v<T, TOther>;
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<TOther>;
        }
    constexpr TensorImpl(TensorImpl<TOther> &&other) noexcept
        : ptr_{other.ptr_}, rank_{other.rank_}, strides_{std::move(other.strides_)}, dims_{std::move(other.dims_)}, size_{other.size_} {
        other.ptr_  = nullptr;
        other.rank_ = 0;
        other.strides_.clear();
        other.dims_.clear();
        other.size_ = 0;
    }

    /**
     * @brief Copy assignment from a datatype with different constness.
     */
    template <typename TOther>
        requires requires {
            requires !std::is_same_v<T, TOther>;
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<TOther>;
        }
    constexpr TensorImpl<T> &operator=(TensorImpl<TOther> const &other) {
        ptr_     = other.ptr_;
        rank_    = other.rank_;
        strides_ = other.strides_;
        dims_    = other.dims_;
        size_    = other.size_;
        return *this;
    }

    /**
     * @brief Move assignment from a datatype with different constness.
     */
    template <typename TOther>
        requires requires {
            requires !std::is_same_v<T, TOther>;
            requires std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>;
            requires std::is_const_v<T> || !std::is_const_v<TOther>;
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

    /**
     * @brief Create a tensor with the given pointer and dimensions.
     *
     * This will calculate the strides using the given memory order.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions of the tensor.
     * @param row_major Whether to compute the strides in row-major or column-major ordering.
     */
    template <Container Dims>
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

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Dims, Container Strides>
    constexpr TensorImpl(pointer ptr, Dims const &dims, Strides const &strides)
        : ptr_{ptr}, rank_{dims.size()}, dims_(dims.begin(), dims.end()), strides_(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>())} {}

    /**
     * @brief Create a tensor with the given pointer and dimensions.
     *
     * This will calculate the strides using the given memory order.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions of the tensor.
     * @param row_major Whether to compute the strides in row-major or column-major ordering.
     */
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, bool row_major = false)
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

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Strides>
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, Strides const &strides)
        : ptr_{ptr}, rank_{dims.size()}, dims_(dims.begin(), dims.end()), strides_(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>())} {}

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Dims>
    constexpr TensorImpl(pointer ptr, Dims const &dims, std::initializer_list<size_t> const &strides)
        : ptr_{ptr}, rank_{dims.size()}, dims_(dims.begin(), dims.end()), strides_(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>())} {}

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, std::initializer_list<size_t> const &strides)
        : ptr_{ptr}, rank_{dims.size()}, dims_(dims.begin(), dims.end()), strides_(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>())} {}

    // Getters and setters.

    /**
     * @brief Get the pointer being wrapped.
     */
    constexpr pointer data() noexcept { return ptr_; }

    /**
     * @brief Get the pointer being wrapped.
     */
    constexpr const_pointer data() const noexcept { return ptr_; }

    /**
     * @brief Get the rank of the tensor.
     */
    constexpr size_t rank() const noexcept { return rank_; }

    /**
     * @brief Get the dimensions of the tensor.
     */
    constexpr BufferVector<size_t> dims() const noexcept { return dims_; }

    /**
     * @brief Get the strides of the tensor.
     */
    constexpr BufferVector<size_t> strides() const noexcept { return strides_; }

    /**
     * @brief Get the size of the tensor.
     */
    constexpr size_t size() const noexcept { return size_; }

    /**
     * @brief Change the pointer being wrapped by the tensor.
     */
    constexpr void set_data(pointer *ptr) noexcept { ptr_ = ptr; }

    // Indexed getters and setters.

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr pointer data(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel_negative_check(strides_, dims_, BufferVector<size_t>{(size_t)index...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr const_pointer data(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel_negative_check(strides_, dims_, BufferVector<size_t>{(size_t)index...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Use only when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr pointer data_no_check(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, BufferVector<size_t>{(size_t)index...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Use only when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr const_pointer data_no_check(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, BufferVector<size_t>{(size_t)index...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
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

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
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

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Use only when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
    constexpr pointer data_no_check(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Use only when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
    constexpr const_pointer data_no_check(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    constexpr pointer data(std::initializer_list<IntType> const &index) {
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

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    constexpr const_pointer data(std::initializer_list<IntType> const &index) const {
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

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Use only when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    constexpr pointer data_no_check(std::initializer_list<IntType> const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Use only when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    constexpr const_pointer data_no_check(std::initializer_list<IntType> const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (ptr_ == nullptr) {
            return nullptr;
        }

        return ptr_ + indices_to_sentinel(strides_, index);
    }

    /**
     * @brief Get the dimension along an axis.
     *
     * Negative values will be wrapped around.
     *
     * @param i The axis to check.
     */
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

        return dims_[temp];
    }

    /**
     * @brief Get the stride along an axis.
     *
     * Negative values will be wrapped around.
     *
     * @param i The axis to check.
     */
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

        return strides_[temp];
    }

    // More complicated getters.

    /**
     * @brief Check whether the tensor is contiguous in memory.
     *
     * For a tensor to be contiguous in memory, there must not be
     * any data outside of any dimension. Views are often not contiguous, though they sometimes can be.
     */
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

    /**
     * @brief Check if a tensor is able to be used as a vector argument to BLAS level 1 calls.
     *
     * @param[out] incx If not nullptr, this will be set to contain the spacing between items.
     */
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
                return dim(-1) * stride(-1) == size_ * stride(0);
            } else {
                if (incx != nullptr) {
                    *incx = stride(-1);
                }
                return dim(0) * stride(0) == size_ * stride(-1);
            }
        }
    }

    /**
     * @brief Checks to see if a tensor can be passed to gemm.
     *
     * For this to be the case, the tensor needs to be rank 2 and its smallest stride must be 1.
     *
     * @param[out] lda The leading dimension which can be passed into gemm and similar calls.
     */
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

    /**
     * @brief Get the smallest stride for the tensor.
     */
    constexpr size_t get_incx() const {
        if (rank_ == 0) {
            return 0;
        } else if (rank_ == 1) {
            return strides_[0];
        } else {
            return std::min(stride(0), stride(-1));
        }
    }

    /**
     * @brief Gets the largest stride for a rank-2 tensor only.
     */
    constexpr size_t get_lda() const {
        if (rank_ != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "Can not get the leading dimension of a tensor whose rank is not 2!");
        } else {
            return std::max(stride(0), stride(1));
        }
    }

    /**
     * @brief Checks to see if a tensor is general row-major.
     *
     * Rank-1 tensors are always both row-major and column-major, so don't assume the two
     * are logically exclusive.
     */
    constexpr bool is_row_major() const {
        if (rank_ <= 1) {
            return true;
        } else {
            return stride(-1) < stride(0);
        }
    }

    /**
     * @brief Checks to see if a tensor is general column-major.
     *
     * Rank-1 tensors are always both row-major and column-major, so don't assume the two
     * are logically exclusive.
     */
    constexpr bool is_column_major() const {
        if (rank_ <= 1) {
            return true;
        } else {
            return stride(-1) > stride(0);
        }
    }

    /**
     * @brief Calculate the parameters for looping over a BLAS call.
     *
     * A quick overview of how this might be used is something like this.
     *
     * @code
     * for(size_t i = 0; i < hard_size; i++) {
     *     blas_call(easy_size, data + i * stride);
     * }
     * @endcode
     *
     * For more complete examples, take a look at the implementation in @c TensorImplOperations.hpp .
     *
     * @param[out] easy_size The number of elements that can be passed into a BLAS call at any given time.
     * @param[out] hard_size The number of times the BLAS call will need to be made.
     * @param[out] easy_rank The largest rank that can be passed into a BLAS call. Any rank less than this
     * can also be passed.
     * @param[out] incx The spacing between elements for the BLAS call.
     */
    void query_vectorable_params(size_t *easy_size, size_t *hard_size, size_t *easy_rank, size_t *incx) const {
        if (rank_ == 0) {
            *easy_size = 0;
            *hard_size = 0;
            *easy_rank = 0;
            *incx      = 0;
        } else if (rank_ == 1) {
            *easy_size = size_;
            *hard_size = 0;
            *easy_rank = 1;
            *incx      = strides_[0];
        } else {

            *easy_size = 1;
            *hard_size = 1;
            *easy_rank = 0;

            if (is_row_major()) {
                *incx       = stride(-1);
                size_t size = 1;
                for (int i = rank_ - 1; i >= 0; i--) {

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

    /**
     * @brief Checks to see if the pointer is associated.
     */
    constexpr bool is_empty_view() const { return ptr_ == nullptr; }

    // Subscript.

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr reference subscript(MultiIndex &&...index) {
        return *data(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr const_reference subscript(MultiIndex &&...index) const {
        return *data(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <Container MultiIndex>
        requires(std::is_integral_v<typename MultiIndex::value_type>)
    constexpr reference subscript(MultiIndex const &index) {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <Container MultiIndex>
        requires(std::is_integral_v<typename MultiIndex::value_type>)
    constexpr const_reference subscript(MultiIndex const &index) const {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr reference subscript_no_check(MultiIndex &&...index) {
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr const_reference subscript_no_check(MultiIndex &&...index) const {
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <Container MultiIndex>
        requires(std::is_integral_v<typename MultiIndex::value_type>)
    constexpr reference subscript_no_check(MultiIndex const &index) {
        return *data_no_check(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <Container MultiIndex>
        requires(std::is_integral_v<typename MultiIndex::value_type>)
    constexpr const_reference subscript_no_check(MultiIndex const &index) const {
        return *data_no_check(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <std::integral IntType>
    constexpr reference subscript(std::initializer_list<IntType> const &index) {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <std::integral IntType>
    constexpr const_reference subscript(std::initializer_list<IntType> const &index) const {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <std::integral IntType>
    constexpr reference subscript_no_check(std::initializer_list<IntType> const &index) {
        return *data_no_check(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <std::integral IntType>
    constexpr const_reference subscript_no_check(std::initializer_list<IntType> const &index) const {
        return *data_no_check(index);
    }

    // Slicing.

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
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

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
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

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr TensorImpl<T> subscript(MultiIndex const &index) {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (ptr_ == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr TensorImpl<std::add_const_t<T>> subscript(MultiIndex const &index) const {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (ptr_ == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    constexpr TensorImpl<T> subscript(std::initializer_list<Range> const &index) {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (ptr_ == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    constexpr TensorImpl<std::add_const_t<T>> subscript(std::initializer_list<Range> const &index) const {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (ptr_ == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
    constexpr TensorImpl<T> subscript_no_check(MultiIndex &&...index) {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

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

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
    constexpr TensorImpl<std::add_const_t<T>> subscript_no_check(MultiIndex &&...index) const {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

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

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr TensorImpl<T> subscript_no_check(MultiIndex const &index) {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (ptr_ == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr TensorImpl<std::add_const_t<T>> subscript_no_check(MultiIndex const &index) const {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (ptr_ == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    constexpr TensorImpl<T> subscript_no_check(std::initializer_list<Range> const &index) {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (ptr_ == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(ptr_ + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    constexpr TensorImpl<std::add_const_t<T>> subscript_no_check(std::initializer_list<Range> const &index) const {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(rank_);
        new_strides.reserve(rank_);

        size_t offset = compute_view(new_dims, new_strides, index);

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

            if (index[1] < index[0] || index[1] > dims_[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Upper bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[1], index[0], dims_[I]);
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

    template <ContainerOrInitializer MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr void adjust_ranges(MultiIndex &indices) const {
        for (auto &[item, dim] : Zip(indices, std::as_const(dims_))) {
            auto temp = item;

            if (item[0] < 0) {
                item[0] += dim;
            }

            if (item[1] < 0) {
                item[1] += dim;
            }

            if (item[0] < 0 || item[0] >= dim) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Lower bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[0], -(ptrdiff_t)dim, dim - 1);
            }

            if (item[1] < item[0] || item[1] > dim) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Upper bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[1], item[0], dim);
            }
        }
    }

    template <ContainerOrInitializer MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr size_t compute_view(BufferVector<size_t> &out_dims, BufferVector<size_t> &out_strides, MultiIndex const &indices) const {
        size_t out = 0;
        for (auto [range, stride] : Zip(indices, strides_)) {

            out_dims.push_back(range[1] - range[0]);
            out_strides.push_back(stride);
            out += range[0] * stride;
        }

        return out;
    }

    pointer              ptr_{nullptr};
    size_t               rank_, size_;
    BufferVector<size_t> dims_, strides_;
};

} // namespace detail
} // namespace einsums