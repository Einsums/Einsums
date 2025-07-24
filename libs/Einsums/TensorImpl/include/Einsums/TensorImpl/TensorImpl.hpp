//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

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
    constexpr TensorImpl() noexcept : _ptr{nullptr}, _dims(), _strides(), _rank{0}, size_{0} {};

    /**
     * @brief Copy constructor.
     */
    constexpr TensorImpl(TensorImpl<T> const &other)
        : _ptr{other._ptr}, _rank{other._rank}, _strides{other._strides}, _dims{other._dims}, size_{other.size_} {}

    /**
     * @brief Move constructor.
     */
    constexpr TensorImpl(TensorImpl<T> &&other) noexcept
        : _ptr{other._ptr}, _rank{other._rank}, _strides{std::move(other._strides)}, _dims{std::move(other._dims)}, size_{other.size_} {
        other._ptr  = nullptr;
        other._rank = 0;
        other._strides.clear();
        other._dims.clear();
        other.size_ = 0;
    }

    /**
     * @brief Copy assignment.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> const &other) {
        _ptr     = other._ptr;
        _rank    = other._rank;
        _strides = other._strides;
        _dims    = other._dims;
        size_    = other.size_;
        return *this;
    }

    /**
     * @brief Move assignment.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> &&other) noexcept {
        _ptr     = other._ptr;
        _rank    = other._rank;
        _strides = std::move(other._strides);
        _dims    = std::move(other._dims);
        size_    = other.size_;

        other._ptr  = nullptr;
        other._rank = 0;
        other._strides.clear();
        other._dims.clear();
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
        : _ptr{other._ptr}, _rank{other._rank}, _strides{other._strides}, _dims{other._dims}, size_{other.size_} {}

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
        : _ptr{other._ptr}, _rank{other._rank}, _strides{std::move(other._strides)}, _dims{std::move(other._dims)}, size_{other.size_} {
        other._ptr  = nullptr;
        other._rank = 0;
        other._strides.clear();
        other._dims.clear();
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
        _ptr     = other._ptr;
        _rank    = other._rank;
        _strides = other._strides;
        _dims    = other._dims;
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
        _ptr     = other._ptr;
        _rank    = other._rank;
        _strides = std::move(other._strides);
        _dims    = std::move(other._dims);
        size_    = other.size_;

        other._ptr  = nullptr;
        other._rank = 0;
        other._strides.clear();
        other._dims.clear();
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
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()) {
        _strides.resize(_rank);

        size_ = 1;

        if (row_major) {
            for (int i = _rank - 1; i >= 0; i--) {
                _strides[i] = size_;
                size_ *= _dims[i];
            }
        } else {
            for (int i = 0; i < _rank; i++) {
                _strides[i] = size_;
                size_ *= _dims[i];
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
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<size_t>())} {}

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
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()) {
        _strides.resize(_rank);

        size_ = 1;

        if (row_major) {
            for (int i = _rank - 1; i >= 0; i--) {
                _strides[i] = size_;
                size_ *= _dims[i];
            }
        } else {
            for (int i = 0; i < _rank; i++) {
                _strides[i] = size_;
                size_ *= _dims[i];
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
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<size_t>())} {}

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Dims>
    constexpr TensorImpl(pointer ptr, Dims const &dims, std::initializer_list<size_t> const &strides)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<size_t>())} {}

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, std::initializer_list<size_t> const &strides)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          size_{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<size_t>())} {}

    // Getters and setters.

    /**
     * @brief Get the pointer being wrapped.
     */
    constexpr pointer data() noexcept { return _ptr; }

    /**
     * @brief Get the pointer being wrapped.
     */
    constexpr const_pointer data() const noexcept { return _ptr; }

    /**
     * @brief Get the rank of the tensor.
     */
    constexpr size_t rank() const noexcept { return _rank; }

    /**
     * @brief Get the dimensions of the tensor.
     */
    constexpr BufferVector<size_t> dims() const noexcept { return _dims; }

    /**
     * @brief Get the strides of the tensor.
     */
    constexpr BufferVector<size_t> strides() const noexcept { return _strides; }

    /**
     * @brief Get the size of the tensor.
     */
    constexpr size_t size() const noexcept { return size_; }

    /**
     * @brief Change the pointer being wrapped by the tensor.
     */
    constexpr void set_data(pointer ptr) noexcept { _ptr = ptr; }

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
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, BufferVector<size_t>{static_cast<size_t>(index)...});
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
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, BufferVector<size_t>{static_cast<size_t>(index)...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr pointer data_no_check(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, BufferVector<size_t>{static_cast<size_t>(index)...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    constexpr const_pointer data_no_check(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, BufferVector<size_t>{static_cast<size_t>(index)...});
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
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
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
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
    constexpr pointer data_no_check(Index const &index) {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
    constexpr const_pointer data_no_check(Index const &index) const {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, index);
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
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
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
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    constexpr pointer data_no_check(std::initializer_list<IntType> const &index) {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    constexpr const_pointer data_no_check(std::initializer_list<IntType> const &index) const {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the dimension along an axis.
     *
     * Negative values will be wrapped around.
     *
     * @param i The axis to check.
     */
    constexpr size_t dim(std::integral auto i) const {
        if (_rank == 0) {
            return 1;
        }
        auto temp = i;
        if (temp < 0) {
            temp += _rank;
        }

        if (temp < 0 || temp >= _rank) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to dim was out of range! Got {}, expected between {} and {}.", i,
                                    -static_cast<ptrdiff_t>(_rank), _rank - 1);
        }

        return _dims[temp];
    }

    /**
     * @brief Get the stride along an axis.
     *
     * Negative values will be wrapped around.
     *
     * @param i The axis to check.
     */
    constexpr size_t stride(std::integral auto i) const {
        if (_rank == 0) {
            return 0;
        }
        auto temp = i;
        if (temp < 0) {
            temp += _rank;
        }

        if (temp < 0 || temp >= _rank) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to stride was out of range! Got {}, expected between {} and {}.",
                                    i, -static_cast<ptrdiff_t>(_rank), _rank - 1);
        }

        return _strides[temp];
    }

    // More complicated getters.

    /**
     * @brief Check whether the tensor is contiguous in memory.
     *
     * For a tensor to be contiguous in memory, there must not be
     * any data outside of any dimension. Views are often not contiguous, though they sometimes can be.
     */
    constexpr bool is_contiguous() const {
        if (_rank == 0) {
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
        if (_rank == 0) {
            if (incx != nullptr) {
                *incx = 0;
            }
            return false;
        } else if (_rank == 1) {
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
        if (_rank != 2) {
            return false;
        } else if (_strides[0] != 1 && _strides[1] != 1) {
            return false;
        } else {
            if (lda != nullptr) {
                *lda = std::max(_strides[0], _strides[1]);
            }
            return true;
        }
    }

    /**
     * @brief Get the smallest stride for the tensor.
     */
    constexpr size_t get_incx() const {
        if (_rank == 0) {
            return 0;
        } else if (_rank == 1) {
            return _strides[0];
        } else {
            return std::min(stride(0), stride(-1));
        }
    }

    /**
     * @brief Gets the largest stride for a rank-2 tensor only.
     */
    constexpr size_t get_lda() const {
        if (_rank != 2) {
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
        if (_rank <= 1) {
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
        if (_rank <= 1) {
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
        if (_rank == 0) {
            *easy_size = 0;
            *hard_size = 0;
            *easy_rank = 0;
            *incx      = 0;
        } else if (_rank == 1) {
            *easy_size = size_;
            *hard_size = 0;
            *easy_rank = 1;
            *incx      = _strides[0];
        } else {

            *easy_size = 1;
            *hard_size = 1;
            *easy_rank = 0;

            if (is_row_major()) {
                *incx       = stride(-1);
                size_t size = 1;
                for (int i = _rank - 1; i >= 0; i--) {

                    if (size * *incx == _strides[i]) {
                        *easy_rank += 1;
                        *easy_size *= _dims[i];
                    } else {
                        *hard_size *= _dims[i];
                    }
                }
            } else {
                *incx       = stride(0);
                size_t size = 1;
                for (int i = 0; i < _rank; i++) {

                    if (size * *incx == _strides[i]) {
                        *easy_rank += 1;
                        *easy_size *= _dims[i];
                    } else {
                        *hard_size *= _dims[i];
                    }
                }
            }
        }
    }

    /**
     * @brief Checks to see if the pointer is associated.
     */
    constexpr bool is_empty_view() const { return _ptr == nullptr; }

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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
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

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<std::add_const_t<T>>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a row-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already row major.
     */
    constexpr TensorImpl<T> to_row_major() {
        if (strides(0) >= strides(-1)) {
            return *this;
        } else {
            return TensorImpl<T>(_ptr, BufferVector<size_t>(_dims.rbegin(), _dims.rend()),
                                 BufferVector<size_t>(_strides.rbegin(), _strides.rend()));
        }
    }

    /**
     * @brief Create a column-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already column major.
     */
    constexpr TensorImpl<T> to_column_major() {
        if (strides(0) <= strides(-1)) {
            return *this;
        } else {
            return TensorImpl<T>(_ptr, BufferVector<size_t>(_dims.rbegin(), _dims.rend()),
                                 BufferVector<size_t>(_strides.rbegin(), _strides.rend()));
        }
    }

    /**
     * @brief Create a row-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already row major.
     */
    constexpr TensorImpl<std::add_const_t<T>> to_row_major() const {
        if (strides(0) >= strides(-1)) {
            return *this;
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr, BufferVector<size_t>(_dims.rbegin(), _dims.rend()),
                                                   BufferVector<size_t>(_strides.rbegin(), _strides.rend()));
        }
    }

    /**
     * @brief Create a column-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already column major.
     */
    constexpr TensorImpl<std::add_const_t<T>> to_column_major() const {
        if (strides(0) <= strides(-1)) {
            return *this;
        } else {
            return TensorImpl<std::add_const_t<T>>(_ptr, BufferVector<size_t>(_dims.rbegin(), _dims.rend()),
                                                   BufferVector<size_t>(_strides.rbegin(), _strides.rend()));
        }
    }

    /**
     * @brief Creates a view with the given indices tied together.
     *
     * This could be useful for Hadamard indices.
     *
     * @param index The index positions to tie together. Negative numbers will be wrapped.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<MultiIndex> && ... && true)
    constexpr TensorImpl<T> tie_indices(MultiIndex &&...index) {
        if constexpr (sizeof...(MultiIndex) <= 1) {
            return *this;
        } else {
            size_t new_stride = 0, new_dim = std::numeric_limits<size_t>::max();

            auto index_array = std::to_array<ptrdiff_t>({static_cast<ptrdiff_t>(index)...});

            BufferVector<size_t> new_strides(_strides), new_dims(_dims);

            // Calculate the tied stride.
            for (size_t i = 0; i < index_array.size(); i++) {
                // Deal with negatives.
                auto temp = index_array[i];
                if (index_array[i] < 0) {
                    index_array[i] += _rank;
                }

                if (index_array[i] < 0 || index_array[i] >= _rank) {
                    EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                            "Attempting to tie indices that are out of bounds! Got {}, expected between {} and {}.", temp,
                                            -static_cast<ptrdiff_t>(_rank), _rank - 1);
                }

                new_stride += _strides[index_array[i]];
                new_strides[index_array[i]] = 0;

                if (_dims[index_array[i]] < new_dim) {
                    new_dim = _dims[index_array[i]];
                }
            }

            // Insert the dim and stride.
            if (is_row_major()) {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride >= _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            } else {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride < _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            }

            // Remove all the zero indices.
            BufferVector<size_t> temp_strides, temp_dims;
            temp_strides.reserve(_rank);
            temp_dims.reserve(_rank);

            for (int i = 0; i < _rank; i++) {
                if (new_strides[i] != 0) {
                    temp_strides.push_back(new_strides[i]);
                    temp_dims.push_back(new_dims[i]);
                }
            }

            return TensorImpl<T>(_ptr, temp_dims, temp_strides);
        }
    }

    /**
     * @brief Creates a view with the given indices tied together.
     *
     * This could be useful for Hadamard indices.
     *
     * @param index The index positions to tie together. Negative numbers will be wrapped.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<MultiIndex> && ... && true)
    constexpr TensorImpl<std::add_const_t<T>> tie_indices(MultiIndex &&...index) const {
        if constexpr (sizeof...(MultiIndex) <= 1) {
            return *this;
        } else {
            size_t new_stride = 0, new_dim = std::numeric_limits<size_t>::max();

            auto index_array = std::to_array<ptrdiff_t>({static_cast<ptrdiff_t>(index)...});

            BufferVector<size_t> new_strides(_strides), new_dims(_dims);

            // Calculate the tied stride.
            for (size_t i = 0; i < index_array.size(); i++) {
                // Deal with negatives.
                auto temp = index_array[i];
                if (index_array[i] < 0) {
                    index_array[i] += _rank;
                }

                if (index_array[i] < 0 || index_array[i] >= _rank) {
                    EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                            "Attempting to tie indices that are out of bounds! Got {}, expected between {} and {}.", temp,
                                            -static_cast<ptrdiff_t>(_rank), _rank - 1);
                }

                // This takes care of duplicates.
                new_stride += new_strides[index_array[i]];
                new_strides[index_array[i]] = 0;

                if (_dims[index_array[i]] < new_dim) {
                    new_dim = _dims[index_array[i]];
                }
            }

            // Insert the dim and stride.
            if (is_row_major()) {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride >= _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            } else {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride < _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            }

            // Remove all the zero strides.
            BufferVector<size_t> temp_strides, temp_dims;
            temp_strides.reserve(_rank);
            temp_dims.reserve(_rank);

            for (int i = 0; i < _rank; i++) {
                if (new_strides[i] != 0) {
                    temp_strides.push_back(new_strides[i]);
                    temp_dims.push_back(new_dims[i]);
                }
            }

            return TensorImpl<std::add_const_t<T>>(_ptr, temp_dims, temp_strides);
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
                index += _dims[I];
            }

            if (index < 0 || index >= _dims[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Index passed to view creation is out of range! Got {}, expected between {} and {}.", temp,
                                        -static_cast<ptrdiff_t>(_dims[I]), _dims[I] - 1);
            }
            adjust_ranges<I + 1>(indices);
        } else if constexpr (std::is_base_of_v<Range, std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
            auto &index = std::get<I>(indices);
            auto  temp  = index;

            if (index[0] < 0) {
                index[0] += _dims[I];
            }

            if (index[1] < 0) {
                index[1] += _dims[I];
            }

            if (index[0] < 0 || index[0] >= _dims[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Lower bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[0], -static_cast<ptrdiff_t>(_dims[I]), _dims[I] - 1);
            }

            if (index[1] < index[0] || index[1] > _dims[I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Upper bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[1], index[0], _dims[I]);
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
                return std::get<I>(indices) * _strides[I] + compute_view<I + 1>(out_dims, out_strides, indices);
            } else if constexpr (std::is_base_of_v<Range, std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
                Range range = std::get<I>(indices);

                out_dims.push_back(range[1] - range[0]);
                out_strides.push_back(_strides[I]);
                return range[0] * _strides[I] + compute_view<I + 1>(out_dims, out_strides, indices);
            } else if constexpr (std::is_base_of_v<AllT, std::tuple_element_t<I, std::remove_cvref_t<decltype(indices)>>>) {
                out_dims.push_back(_dims[I]);
                out_strides.push_back(_strides[I]);
                return compute_view<I + 1>(out_dims, out_strides, indices);
            }
        }
    }

    template <ContainerOrInitializer MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr void adjust_ranges(MultiIndex &indices) const {
        for (auto &[item, dim] : Zip(indices, std::as_const(_dims))) {
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
                                        temp[0], -static_cast<ptrdiff_t>(dim), dim - 1);
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
        for (auto [range, stride] : Zip(indices, _strides)) {

            out_dims.push_back(range[1] - range[0]);
            out_strides.push_back(stride);
            out += range[0] * stride;
        }

        return out;
    }

    pointer              _ptr{nullptr};
    size_t               _rank, size_;
    BufferVector<size_t> _dims, _strides;
};

} // namespace detail
} // namespace einsums