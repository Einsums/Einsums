//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>
#include <Einsums/Utilities/Tuple.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <concepts>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "Einsums/Config/CompilerSpecific.hpp"
#include "Einsums/Tensor/DeviceTensor.hpp"
#include "Einsums/TensorImpl/TensorImplOperations.hpp"

#if defined(EINSUMS_COMPUTE_CODE)
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {
#ifndef DOXYGEN
// Forward declaration of the Tensor printing function.
template <RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void println(AType const &A, TensorPrintOptions options = {});

template <FileOrOStream Output, RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void fprintln(Output &fp, AType const &A, TensorPrintOptions options = {});
#endif

/**
 * @brief Represents a general tensor
 *
 * @tparam T data type of the underlying tensor data
 * @tparam Rank the rank of the tensor
 */
template <typename T, size_t rank, typename Alloc>
struct GeneralTensor : tensor_base::CoreTensor, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {
    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = T *;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = T const *;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = T &;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = T const &;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef Vector
     *
     * This represents the internal storage method of the tensor.
     */
    using Vector = std::vector<T, Alloc>;

    using Allocator = Alloc;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    GeneralTensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    GeneralTensor(GeneralTensor const &other) : _name(other.name()), _data(other._data), _impl(other._impl) {
        _impl.set_data(_data.data());
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object from one with a different allocator.
     */
    template <typename Alloc2>
    GeneralTensor(GeneralTensor<T, Rank, Alloc2> const &other) : _name(other.name()), _data(other._data), _impl(other._impl) {
        _impl.set_data(_data.data());
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Destroy the Tensor object.
     */
    ~GeneralTensor() = default;

    /**
     * @brief Default move constructor.
     */
    GeneralTensor(GeneralTensor &&other) noexcept
        : _name{std::move(other._name)}, _data{std::move(other._data)}, _dim_array{std::move(other._dim_array)},
          _stride_array{std::move(other._stride_array)}, _impl{std::move(other._impl)} {}

    /**
     * @brief Default move assignment.
     */
    GeneralTensor &operator=(GeneralTensor &&other) noexcept {
        _name         = std::move(other._name);
        _data         = std::move(other._data);
        _dim_array    = std::move(other._dim_array);
        _stride_array = std::move(other._stride_array);
        _impl         = std::move(other._impl);

        return *this;
    }

    /**
     * @brief Construct a new Tensor object with the given name and dimensions.
     *
     * Constructs a new Tensor object using the information provided in \p name and \p dims .
     *
     * @code
     * auto A = Tensor("A", 3, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param dims The dimensions of each rank of the tensor.
     */
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    GeneralTensor(std::string name, Dims... dims)
        : _name{std::move(name)}, _impl(nullptr, std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(dims)...}, row_major_default) {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        // Resize the data structure
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object with the given name and dimensions.
     *
     * Constructs a new Tensor object using the information provided in \p name and \p dims .
     *
     * @code
     * auto A = Tensor("A", 3, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param dims The dimensions of each rank of the tensor.
     */
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    GeneralTensor(bool row_major, std::string name, Dims... dims)
        : _name{std::move(name)}, _impl(nullptr, std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(dims)...}, row_major) {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        // Resize the data structure
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object. Moving \p existingTensor data to the new tensor.
     *
     * This constructor is useful for reshaping a tensor. It does not modify the underlying
     * tensor data. It only creates new mapping arrays for how the data is viewed.
     *
     * @code
     * auto A = Tensor("A", 27); // Creates a rank-1 tensor of 27 elements
     * auto B = Tensor(std::move(A), "B", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     * // At this point A is no longer valid.
     * @endcode
     *
     * Supports using -1 for one of the ranks to automatically compute the dimension of it.
     *
     * @code
     * auto A = Tensor("A", 27);
     * auto B = Tensor(std::move(A), "B", 3, -1, 3); // Automatically determines that -1 should be 3.
     * @endcode
     *
     * @tparam OtherRank The rank of \p existingTensor can be different from the rank of the new tensor
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param existingTensor The existing tensor that holds the tensor data.
     * @param name The name of the new tensor
     * @param dims The dimensionality of each rank of the new tensor.
     */
    template <size_t OtherRank, typename... Dims>
    explicit GeneralTensor(GeneralTensor<T, OtherRank, Alloc> &&existingTensor, std::string name, Dims... dims)
        : _name{std::move(name)}, _data(std::move(existingTensor._data)) {
        static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we
        // compute this dimensionality of this "0" index for them.

        auto _dims = std::array<ptrdiff_t, sizeof...(Dims)>{(ptrdiff_t)dims...};

        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(_dims)) {
            if (dim == -1) {
                nfound++;
                location = i;
            }
        }

        if (nfound > 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "More than one -1 was provided.");
        }

        if (nfound == 1) {
            size_t size{1};
            for (auto [i, dim] : enumerate(_dims)) {
                if (i != location) {
                    size *= dim;
                }
            }
            if (size > existingTensor.size()) {
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Size of new tensor is larger than the parent tensor.");
            }
            _dims[location] = existingTensor.size() / size;
        }

        _impl = detail::TensorImpl<T>(_data.data(), _dims, existingTensor.impl().is_row_major());

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        // Check size
        if (_data.size() != _impl.size()) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Provided dims to not match size of parent tensor");
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit GeneralTensor(Dim<Rank> dims) : _impl(nullptr, dims) {
        // Resize the data structure
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit GeneralTensor(bool row_major, Dim<Rank> dims) : _impl(nullptr, dims, row_major) {
        // Resize the data structure
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor object from a TensorView.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     */
    GeneralTensor(TensorView<T, rank> const &other) : _name{other.name()}, _impl(nullptr, other.dims(), row_major_default) {
        // Resize the data structure
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Construct a new Tensor from the implementation of another.
     */
    GeneralTensor(detail::TensorImpl<T> const &other) : _impl(nullptr, other.dims()) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
        detail::copy_to(other, _impl);

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    void resize(Dim<Rank> dims) {
        if (std::equal(_impl.dims().cbegin(), _impl.dims().cend(), dims.cbegin())) {
            return;
        }

        _impl = detail::TensorImpl<T>(nullptr, dims, _impl.is_row_major());

        // Resize the data structure
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    template <std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    void resize(Dims... dims) {
        resize(Dim<Rank>{static_cast<size_t>(dims)...});
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() { memset(_data.data(), 0, sizeof(T) * _data.size()); }

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) { std::fill(_data.begin(), _data.end(), value); }

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return T* A pointer to the data.
     */
    Pointer data() { return _impl.data(); }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    ConstPointer data() const { return _impl.data(); }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     * @code
     * auto A = Tensor("A", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     *
     * double* A_pointer = A.data(1, 2, 3); // Returns the pointer to element (1, 2, 3) in A.
     * @endcode
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to ptrdiff_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto data(MultiIndex &&...index) -> Pointer {
#if !defined(DOXYGEN)
        assert(sizeof...(MultiIndex) <= Rank);

        return _impl.data(std::forward<MultiIndex>(index)...);
#endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to ptrdiff_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to ptrdiff_t.
     *              Negative indices are taken to be an offset from the end of the axis.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam int_type The type of integer used for the indices.
     * @param index The explicit desired index into the tensor. Elements must be castable size_t.
     * @return const T&
     */
    template <Container Index>
    auto subscript(Index const &index) const -> ConstReference {
        return _impl.subscript_no_check(index);
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to size_t.
     * @param index The explicit desired index into the tensor. Elements must be castable size_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to ptrdiff_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to ptrdiff_t.
     *              Negative indices are taken to be an offset from the end of the axis.
     * @return T&
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam int_type The type of integer used for the indices.
     * @param index The explicit desired index into the tensor. Elements must be castable size_t.
     * @return T&
     */
    template <Container Index>
    auto subscript(Index const &index) -> Reference {
        return _impl.subscript_no_check(index);
    }

    /**
     * @brief Subscripts into the tensor. Does not do any index checks.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to ptrdiff_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to ptrdiff_t.
     * @return T&
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when some of the indices are All or Range. It then constructs a view of the tensor with those properties.
     *
     * @tparam MultiIndex Data type of the indices.
     * @param index The indices.
     * @return A tensor view with the appropriate starting point and dimensions.
     */
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>; }
    auto operator()(MultiIndex const &...index)
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        // println("Here");

        return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{
            _impl.template subscript<true>(index...), data()};
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when some of the indices are All or Range. It then constructs a view of the tensor with those properties.
     *
     * @tparam MultiIndex Data type of the indices.
     * @param index The indices.
     * @return A tensor view with the appropriate starting point and dimensions.
     */
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>; }
    auto operator()(MultiIndex const &...index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??

        return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(
            _impl.template subscript<true>(index...));
    }

    /**
     * @copydoc GeneralTensor<T, Rank, Alloc>::operator(MultiIndex...) -> T&
     */
    template <Container Index>
        requires requires {
            requires !std::is_same_v<Index, Offset<Rank>>;
            requires !std::is_same_v<Index, Range>;
        }
    Reference operator()(Index const &index) {
        return _impl.subscript(index);
    }

    /**
     * @copydoc GeneralTensor<T, Rank, Alloc>::operator(MultiIndex...) -> T&
     */
    template <Container Index>
        requires requires {
            requires !std::is_same_v<Index, Offset<Rank>>;
            requires !std::is_same_v<Index, Range>;
        }
    ConstReference operator()(Index const &index) const {
        return _impl.subscript(index);
    }

    /**
     * Copy the data from one tensor into this.
     *
     * @param other The tensor to copy.
     */
    auto operator=(GeneralTensor const &other) -> GeneralTensor & {
        LabeledSection("operator=");
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = other.impl();

            _data.resize(_impl.size());

            _impl.set_data(_data.data());
        }

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    template <typename Alloc2>
    auto operator=(GeneralTensor<T, Rank, Alloc2> const &other) -> GeneralTensor & {
        LabeledSection("operator=");
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = other.impl();

            _data.resize(_impl.size());

            _impl.set_data(_data.data());
        }

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * Cast the data from one tensor while copying its data into this tensor.
     *
     * @param other The tensor to cast and copy.
     */
    template <typename TOther, typename Alloc2>
        requires(!std::same_as<T, TOther>)
    auto operator=(GeneralTensor<TOther, Rank, Alloc2> const &other) -> GeneralTensor & {
        LabeledSection("operator=");
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = detail::TensorImpl<T>(nullptr, other.dims(), other.impl().is_row_major());

            _data.resize(_impl.size());

            _impl.set_data(_data.data());
        }

        detail::copy_to(other.impl(), _impl);

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        return *this;
    }

    /**
     * Copy the data from a tensor of a different kind into this one.
     */

    template <BasicTensorConcept OtherTensor>
    auto operator=(OtherTensor const &other) -> GeneralTensor & {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    template <TensorConcept OtherTensor>
        requires requires {
            requires !BasicTensorConcept<OtherTensor>;
            requires SameRank<GeneralTensor, OtherTensor>;
            requires CoreTensorConcept<OtherTensor>;
        }
    auto operator=(OtherTensor const &other) -> GeneralTensor & {
        LabeledSection("operator=");
        size_t size = this->size();

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < size; sentinel++) {
            thread_local std::array<size_t, Rank> index;
            sentinel_to_indices(sentinel, _impl.strides(), index);
            _data[sentinel] = subscript_tensor(other, index);
        }

        return *this;
    }

    /**
     * Cast the data from a tensor view while copying into this tensor.
     */
    template <typename TOther>
    auto operator=(TensorView<TOther, Rank> const &other) -> GeneralTensor & {
        LabeledSection("operator=");
    detail:
        copy_to(other.impl(), _impl);

        return *this;
    }

#ifdef EINSUMS_COMPUTE_CODE
    /**
     * Copy the data from the device into this tensor.
     */
    auto operator=(DeviceTensor<T, Rank> const &other) -> GeneralTensor<T, Rank, Alloc> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i))) {
                realloc = true;
            }
        }

        if (realloc) {
            _impl = detail::TensorImpl<T>(nullptr, other.dims());

            // Resize the data structure
            _data.resize(size());

            _impl.set_data(_data.data());
        }

        hip_catch(hipMemcpy(_data.data(), other.gpu_data(), _impl.size() * sizeof(T), hipMemcpyDeviceToHost));

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }

        return *this;
    }
#endif

    /**
     * Fill this tensor with a value.
     */
    auto operator=(T const &fill_value) -> GeneralTensor & {
        LabeledSection("operator= value");
        set_all(fill_value);
        return *this;
    }

    GeneralTensor &operator+=(T const &b) {
        detail::impl_scalar_add_contiguous(b, _impl);

        return *this;
    }

    GeneralTensor &operator-=(T const &b) {
        detail::impl_scalar_add_contiguous(-b, _impl);

        return *this;
    }

    GeneralTensor &operator*=(T const &b) {
        detail::impl_scal_contiguous(b, _impl);

        return *this;
    }

    GeneralTensor &operator/=(T const &b) {
        detail::impl_div_scalar_contiguous(b, _impl);

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator+=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy_contiguous(T{1}, other.impl(), _impl);
            } else {
                detail::impl_axpy_contiguous(T{1.0}, other.impl(), _impl);
            }
        } else {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy(T{1}, other.impl(), _impl);
            } else {
                detail::impl_axpy(T{1.0}, other.impl(), _impl);
            }
        }

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator-=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy_contiguous(T{-1}, other.impl(), _impl);
            } else {
                detail::impl_axpy_contiguous(T{-1.0}, other.impl(), _impl);
            }
        } else {
            if constexpr (std::is_integral_v<T>) {
                detail::impl_axpy(T{-1}, other.impl(), _impl);
            } else {
                detail::impl_axpy(T{-1.0}, other.impl(), _impl);
            }
        }

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator*=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            detail::impl_mult_contiguous(other.impl(), _impl);
        } else {
            detail::impl_mult(other.impl(), _impl);
        }

        return *this;
    }

    template <typename TOther, typename Alloc2>
    GeneralTensor &operator/=(GeneralTensor<TOther, rank, Alloc2> const &other) {
        if (_impl.is_column_major() == other.impl().is_column_major()) {
            detail::impl_div_contiguous(other.impl(), _impl);
        } else {
            detail::impl_div(other.impl(), _impl);
        }

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator+=(TOther const &other) {
        detail::add_assign(other.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator-=(TOther const &other) {
        detail::sub_assign(other.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator*=(TOther const &other) {
        detail::mult_assign(other.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    GeneralTensor &operator/=(TOther const &other) {
        detail::div_assign(other.impl(), _impl);

        return *this;
    }

    /**
     * Get the dimension of the tensor along a given axis.
     */
    size_t dim(int d) const { return _impl.dim(d); }

    /**
     * Get all the dimensions of the tensor.
     */
    Dim<Rank> const &dims() const { return _dim_array; }

    /**
     * Get the internal vector containing the tensor's data.
     */
    auto vector_data() const -> Vector const & { return _data; }

    /// @copydoc GeneralTensor<T,Rank, Alloc>::vector_data() const
    auto vector_data() -> Vector & { return _data; }

    /**
     * Get the stride along a given axis.
     */
    size_t stride(int d) const { return _impl.stride(d); }

    /**
     * Get the strides of this tensor.
     */
    Stride<Rank> const &strides() const { return _stride_array; }

    /**
     * Flatten out the tensor.
     */
    auto to_rank_1_view() const -> TensorView<T, 1> {
        Dim<1> dim{size()};

        return TensorView<T, 1>{*this, dim};
    }

    /**
     * Returns the linear size of the tensor.
     */
    [[nodiscard]] size_t size() const { return _impl.size(); }

    /**
     * Indicates that the tensor is contiguous.
     */
    bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Get the name of the tensor.
     */
    std::string const &name() const { return _name; };

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; };

    /**
     * Get the implementation of the tensor.
     */
    detail::TensorImpl<T> &impl() { return _impl; }

    /**
     * Get the implementation of the tensor.
     */
    detail::TensorImpl<T> const &impl() const { return _impl; }

    bool is_row_major() const { return _impl.is_row_major(); }

    bool is_column_major() const { return _impl.is_column_major(); }

    TensorView<T, Rank> transpose_view() { return TensorView<T, Rank>(_impl.transpose_view()); }

    TensorView<T, Rank> const transpose_view() const { return TensorView<T, Rank>(_impl.transpose_view()); }

    TensorView<T, Rank> to_row_major() { return TensorView<T, Rank>(_impl.to_row_major()); }

    TensorView<T, Rank> const to_row_major() const { return TensorView<T, Rank>(_impl.to_row_major()); }

    TensorView<T, Rank> to_column_major() { return TensorView<T, Rank>(_impl.to_column_major()); }

    TensorView<T, Rank> const to_column_major() const { return TensorView<T, Rank>(_impl.to_column_major()); }

    bool is_gemmable(size_t *lda = nullptr) const { return _impl.is_gemmable(lda); }

    bool is_totally_vectorable(size_t *incx = nullptr) const { return _impl.is_totally_vectorable(incx); }

    template <std::integral... MultiIndex>
    TensorView<T, Rank - sizeof...(MultiIndex) + 1> tie_indices(MultiIndex &&...index) {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    TensorView<T, Rank - sizeof...(MultiIndex) + 1> const tie_indices(MultiIndex &&...index) const {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    void tensor_to_gpu() const { _impl.tensor_to_gpu(); }

    void tensor_from_gpu() { _impl.tensor_from_gpu(); }

    auto gpu_cache_tensor() { return _impl.gpu_cache_tensor(); }

    auto gpu_cache_tensor_nowrite() { return _impl.gpu_cache_tensor_nowrite(); }

    auto gpu_cache_tensor() const { return _impl.gpu_cache_tensor(); }

    auto gpu_cache_tensor_nowrite() const { return _impl.gpu_cache_tensor_nowrite(); }

    auto get_gpu_pointer() { return _impl.get_gpu_pointer(); }

    auto get_gpu_pointer() const { return _impl.get_gpu_pointer(); }

    auto get_gpu_memory() const { return _impl.get_gpu_memory(); }

    bool gpu_is_expired() const { return _impl.gpu_is_expired(); }

  private:
    std::string _name{"(unnamed)"};

    Vector _data{};

    detail::TensorImpl<T> _impl{};

    Dim<Rank>    _dim_array;
    Stride<Rank> _stride_array;

    template <typename T_, size_t Rank_>
    friend struct TensorView;

    template <typename T_, size_t OtherRank, typename Alloc2>
    friend struct GeneralTensor;
};

/**
 * @struct Tensor<T, 0>
 *
 * @brief Represents a zero-rank tensor. It has a special implementation since it is essentially a scalar.
 *
 * @tparam T The data type being stored.
 */
template <typename T, typename Alloc>
struct GeneralTensor<T, 0, Alloc> final : tensor_base::CoreTensor,
                                          design_pats::Lockable<std::recursive_mutex>,
                                          tensor_base::AlgebraOptimizedTensor {

    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = 0;

    /**
     * Default constructor
     */
    GeneralTensor() = default;

    /**
     * Default copy constructor
     */
    GeneralTensor(GeneralTensor const &) = default;

    /**
     * Default move constructor
     */
    GeneralTensor(GeneralTensor &&) noexcept = default;

    /**
     * Default destructor
     */
    ~GeneralTensor() = default;

    /**
     * Create a new zero-rank tensor with the given name.
     */
    explicit GeneralTensor(std::string name) : _name{std::move(name)} {};

    /**
     * Create a new zero-rank tensor with the given dimensions. Since it is zero-rank,
     * the dimensions will be empty, and are ignored.
     */
    explicit GeneralTensor(Dim<0> _ignore) {}

    /**
     * Get the pointer to the data stored by this tensor.
     */
    T *data() { return &_data; }

    /**
     * @copydoc GeneralTensor<T,0,Alloc>::data()
     */
    T const *data() const { return &_data; }

    /**
     * Copy assignment.
     */
    auto operator=(GeneralTensor const &other) -> GeneralTensor & {
        _data = other._data;
        return *this;
    }

    /**
     * Set the value of the tensor to the value passed in.
     */
    auto operator=(T const &other) -> GeneralTensor & {
        _data = other;
        return *this;
    }

#ifndef DOXYGEN
#    if defined(OPERATOR)
#        undef OPERATOR
#    endif
#    define OPERATOR(OP)                                                                                                                   \
        auto operator OP(const T &other)->GeneralTensor & {                                                                                \
            _data OP other;                                                                                                                \
            return *this;                                                                                                                  \
        }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#    undef OPERATOR
#endif

    /**
     * Cast the tensor to a scalar.
     */
    operator T() const { return _data; } // NOLINT

    /**
     * Cast the tensor to a scalar.
     */
    operator T &() { return _data; } // NOLINT

    /**
     * Get the name of the tensor.
     */
    std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * Get the dimension of the tensor. Always returns 1.
     */
    size_t dim(int) const { return 1; }

    /**
     * Get the dimensions of the tensor. The result is empty.
     */
    Dim<0> dims() const { return Dim{}; }

    /**
     * Indicates that the tensor is contiguous.
     */
    bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Get the stride of the tensor. Always returns 1.
     */
    size_t stride(int d) const { return 0; }

    /**
     * Get the strides of the tensor. The result is empty.
     */
    Stride<0> strides() const { return Stride{}; }

  private:
    /**
     * @var _name
     *
     * The name of the tensor used for printing.
     */
    std::string _name{"(Unnamed)"};

    /**
     * The value stored by the tensor.
     */
    T _data{};
};

/**
 * @struct TensorView
 *
 * @brief Represents a view of a tensor, which may have different dimensions and start at a different index.
 *
 * @tparam T The data type being stored.
 * @tparam Rank The rank of the view.
 */
template <typename T, size_t rank>
struct TensorView final : tensor_base::CoreTensor, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {
    /**
     * @typedef ValueType
     *
     * @brief Holds the data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = T *;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = T const *;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = T &;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = T const &;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor view.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef underlying_type
     *
     * @brief The type of tensor this view views.
     */
    using underlying_type = Tensor<T, Rank>;

    TensorView() = delete;

    /**
     * Default copy constructor.
     */
    TensorView(TensorView const &) = default;

    /**
     * Default destructor.
     */
    ~TensorView() = default;

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and TensorView).  The
    // call to common_initialization is able to perform an enable_if check.
    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename Alloc, typename... Args>
    explicit TensorView(GeneralTensor<T, OtherRank, Alloc> const &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(const_cast<GeneralTensor<T, OtherRank, Alloc> &>(other), dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename Alloc, typename... Args>
    explicit TensorView(GeneralTensor<T, OtherRank, Alloc> &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(other, dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(other, dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> const &other, Dim const &dims, Args &&...args) : _name{other._name} {
        common_initialization(const_cast<TensorView<T, OtherRank> &>(other), dims, std::forward<Args>(args)...);
    }

    /**
     * Creates a view of a tensor with the given properties.
     */
    template <size_t OtherRank, Container Dim, typename Alloc, typename... Args>
    explicit TensorView(std::string name, GeneralTensor<T, OtherRank, Alloc> &other, Dim const &dims, Args &&...args)
        : _name{std::move(name)} {
        common_initialization(other, dims, std::forward<Args>(args)...);
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims, bool row_major = row_major_default)
        : _impl(const_cast<T *>(data), dims, row_major), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a pointer in a tensor view, specifying the dimensions.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     */
    explicit TensorView(T *data, Dim<Rank> const &dims, bool row_major = row_major_default)
        : _impl(const_cast<T *>(data), dims, row_major), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions and strides.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     * @param strides The strides for the view.
     */
    explicit TensorView(T const *data, Dim<Rank> const &dims, Stride<Rank> const &strides)
        : _impl(const_cast<T *>(data), dims, strides), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Wrap a const pointer in a tensor view, specifying the dimensions and strides.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the view.
     * @param strides The strides for the view.
     */
    explicit TensorView(T *data, Dim<Rank> const &dims, Stride<Rank> const &strides)
        : _impl(const_cast<T *>(data), dims, strides), _parent{const_cast<T *>(data)} {
        _offsets.fill(0);
        _source_dims = dims;
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    explicit TensorView(detail::TensorImpl<T> const &impl) : _impl(impl), _parent{const_cast<T *>(impl.data())} {
        _offsets.fill(0);
        _source_dims = Dim<Rank>(impl.dims().begin(), impl.dims().end());
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }
    explicit TensorView(detail::TensorImpl<T> const &impl, T *parent) : _impl(impl), _parent{parent} {
        _offsets.fill(0);
        _source_dims = Dim<Rank>(impl.dims().begin(), impl.dims().end());
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    explicit TensorView(detail::TensorImpl<T> const &impl, T const *parent) : _impl(impl), _parent{const_cast<T *>(parent)} {
        _offsets.fill(0);
        _source_dims = Dim<Rank>(impl.dims().begin(), impl.dims().end());
        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * Copy data from a pointer into this view. It always assumes that the source array is stored in row-major
     * order.
     *
     * @attention This is an expert function only. If you are using it, you must know what you are doing!
     */
    auto operator=(T const *other) -> TensorView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        detail::TensorImpl<T> other_impl(const_cast<T *>(other), _impl.dims(), true);

        detail::copy_to(other_impl, _impl);

        return *this;
    }

    /**
     * Copy assignment.
     */
    auto operator=(TensorView const &other) -> TensorView & {
        if (this == &other)
            return *this;

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * Copy the data from another tensor into this view.
     */
    template <typename AType>
        requires CoreRankTensor<AType, Rank, T>
    auto operator=(AType const &other) -> TensorView & {
        if constexpr (std::is_same_v<AType, TensorView>) {
            if (this == &other)
                return *this;
        }

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * Fill this view with a single value.
     */
    auto operator=(T const &fill_value) -> TensorView & {
        detail::copy_to(fill_value, _impl);

        return *this;
    }

    void zero() { *this = T{0.0}; }

    void set_all(T value) { *this = value; }

    template <typename TOther>
    TensorView &operator+=(TOther const &b) {
        detail::add_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    TensorView &operator-=(TOther const &b) {
        detail::sub_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    TensorView &operator*=(TOther const &b) {
        detail::mult_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    TensorView &operator/=(TOther const &b) {
        detail::div_assign(b, _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator+=(TOther const &b) {
        detail::add_assign(b.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator-=(TOther const &b) {
        detail::sub_assign(b.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator*=(TOther const &b) {
        detail::mult_assign(b.impl(), _impl);

        return *this;
    }

    template <BasicTensorConcept TOther>
    TensorView &operator/=(TOther const &b) {
        detail::div_assign(b.impl(), _impl);

        return *this;
    }

    /**
     * Get a pointer to the data.
     */
    Pointer data() { return _impl.data(); }

    /**
     * @copydoc TensorView<T,Rank>::data()
     */
    ConstPointer data() const { return _impl.data(); }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index The index for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto data(MultiIndex &&...index) const -> ConstPointer {
        return _impl.data(std::forward<MultiIndex>(index)...);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    template <Container Index>
    auto data(Index const &index_list) const -> ConstPointer {
        return _impl.data(index_list);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index The index for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto data(MultiIndex &&...index) -> Pointer {
        return _impl.data(std::forward<MultiIndex>(index)...);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    template <Container Index>
    auto data(Index const &index_list) -> Pointer {
        return _impl.data(index_list);
    }

    /**
     * Get a pointer to the data at a certain index in the tensor.
     *
     * @param index_list The index for the offset.
     */
    template <Container Index>
    [[deprecated("The data_array method will be removed in the future. Its functionality will be taken by data.")]] auto
    data_array(Index const &index_list) const -> Pointer {
        return const_cast<Pointer>(data(index_list));
    }

    Pointer full_data() noexcept { return _parent; }

    ConstPointer full_data() const noexcept { return _parent; }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto operator()(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, std::remove_cvref_t<MultiIndex>...> || AtLeastOneOfType<Range, std::remove_cvref_t<MultiIndex>...>)
    auto operator()(MultiIndex &&...index) -> TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                                                count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()> {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>)
    auto operator()(MultiIndex &&...index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>)
    auto subscript(MultiIndex &&...index) -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    template <typename... MultiIndex>
        requires(AtLeastOneOfType<AllT, MultiIndex...> || AtLeastOneOfType<Range, MultiIndex...>)
    auto subscript(MultiIndex &&...index) const
        -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        static_assert(sizeof...(MultiIndex) == Rank);

        return TensorView<T, count_of_type<AllT, std::remove_cvref_t<MultiIndex>...>() +
                                 count_of_type<Range, std::remove_cvref_t<MultiIndex>...>()>(
            _impl.template subscript<true>(std::forward<MultiIndex>(index)...), _parent);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) const -> ConstReference {
        static_assert(sizeof...(MultiIndex) == Rank);
        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...)
    auto subscript(MultiIndex &&...index) -> Reference {
        static_assert(sizeof...(MultiIndex) == Rank);

        return _impl.subscript_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto subscript(std::array<int_type, Rank> const &index) const -> ConstReference {
        return _impl.subscript_no_check(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Does not do any checking of the indices.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto subscript(std::array<int_type, Rank> const &index) -> Reference {
        return _impl.subscript_no_check(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    Reference operator()(Container const &index) {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * Wraps negative indices and checks the bounds.
     *
     * @param index The indices to use for the subscript.
     */
    template <typename Container>
        requires requires {
            requires !std::is_integral_v<Container>;
            requires !std::is_same_v<Container, Dim<Rank>>;
            requires !std::is_same_v<Container, Stride<Rank>>;
            requires !std::is_same_v<Container, Offset<Rank>>;
            requires !std::is_same_v<Container, Range>;
        }
    ConstReference operator()(Container const &index) const {
        return _impl.subscript(index);
    }

    /**
     * Get the dimension of the view along a given axis.
     *
     * @param d The axis to query.
     */
    size_t dim(int d) const { return _impl.dim(d); }

    /**
     * Get the dimension of the original tensor along a given axis.
     */
    size_t source_dim(int d) const {
        int temp = d;
        if (temp < 0) {
            temp += Rank;
        }

        if (temp < 0 || temp >= Rank) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index to the source dims was out of range! Expected between {} and {}, got {}.",
                                    -(ptrdiff_t)Rank, Rank - 1, d);
        }

        return _source_dims[temp];
    }

    /**
     * Get the dimensions of the view.
     */
    Dim<Rank> const &dims() const { return _dim_array; }

    /**
     * Get the dimensions of the original tensor.
     */
    Dim<Rank> const &source_dims() const { return _source_dims; }

    /**
     * Get the name of the view.
     */
    std::string const &name() const { return _name; }

    /**
     * Set the name of the view.
     *
     * @param name The new name for the view.
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * Get the stride of the view along a given axis.
     *
     * @param d The axis to query.
     */
    size_t stride(int d) const { return _impl.stride(d); }

    /**
     * Get the strides of the tensor.
     */
    Stride<Rank> const &strides() const noexcept { return _stride_array; }

    /**
     * Get the offset of the view along a given axis.
     */
    size_t offset(int d) const {
        int temp = d;
        if (temp < 0) {
            temp += Rank;
        }

        if (temp < 0 || temp >= Rank) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index to the offsets was out of range! Expected between {} and {}, got {}.",
                                    -(ptrdiff_t)Rank, Rank - 1, d);
        }

        return _offsets[temp];
    }

    /**
     * Get the offsets of the tensor.
     */
    Offset<Rank> offsets() const noexcept { return _offsets; }

    /**
     * Flatten the view.
     *
     * @warning Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!
     */
    auto to_rank_1_view() const -> TensorView<T, 1> {
        if constexpr (Rank == 1) {
            return *this;
        } else {
            if (std::min(_impl.stride(0), _impl.stride(-1)) != 1) {
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Creating a Rank-1 TensorView for this Tensor(View) is not supported.");
            }
            size_t size = Rank == 0 ? 0 : std::max(stride(0) * dim(0), stride(-1) * dim(-1));
            Dim<1> dim{size};

#if defined(EINSUMS_SHOW_WARNING)
            println("Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!");
#endif

            return TensorView<T, 1>{*this, dim, Stride{1}};
        }
    }

    /**
     * Check whether the view has all elements of the tensor it is viewing.
     */
    bool full_view_of_underlying() const noexcept { return _impl.is_contiguous(); }

    /**
     * Get the number of elements in the view.
     */
    size_t size() const { return _impl.size(); }

    /**
     * Get the underlying implementation details.
     */
    detail::TensorImpl<T> &impl() noexcept { return _impl; }

    /**
     * Get the underlying implementation details.
     */
    detail::TensorImpl<T> const &impl() const noexcept { return _impl; }

    bool is_row_major() const { return _impl.is_row_major(); }

    bool is_column_major() const { return _impl.is_column_major(); }

    TensorView<T, Rank> transpose_view() { return TensorView<T, Rank>(_impl.transpose_view()); }

    TensorView<T, Rank> const transpose_view() const { return TensorView<T, Rank>(_impl.transpose_view()); }

    TensorView<T, Rank> to_row_major() { return TensorView<T, Rank>(_impl.to_row_major()); }

    TensorView<T, Rank> const to_row_major() const { return TensorView<T, Rank>(_impl.to_row_major()); }

    TensorView<T, Rank> to_column_major() { return TensorView<T, Rank>(_impl.to_column_major()); }

    TensorView<T, Rank> const to_column_major() const { return TensorView<T, Rank>(_impl.to_column_major()); }

    bool is_gemmable(size_t *lda = nullptr) const { return _impl.is_gemmable(lda); }

    bool is_totally_vectorable(size_t *incx = nullptr) const { return _impl.is_totally_vectorable(incx); }

    template <std::integral... MultiIndex>
    TensorView<T, Rank - sizeof...(MultiIndex) + 1> tie_indices(MultiIndex &&...index) {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    TensorView<T, Rank - sizeof...(MultiIndex) + 1> const tie_indices(MultiIndex &&...index) const {
        return TensorView<T, Rank - sizeof...(MultiIndex) + 1>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    void tensor_to_gpu() const { _impl.tensor_to_gpu(); }

    void tensor_from_gpu() { _impl.tensor_from_gpu(); }

    auto gpu_cache_tensor() { return _impl.gpu_cache_tensor(); }

    auto gpu_cache_tensor_nowrite() { return _impl.gpu_cache_tensor_nowrite(); }

    auto gpu_cache_tensor() const { return _impl.gpu_cache_tensor(); }

    auto gpu_cache_tensor_nowrite() const { return _impl.gpu_cache_tensor_nowrite(); }

    auto get_gpu_pointer() { return _impl.get_gpu_pointer(); }

    auto get_gpu_pointer() const { return _impl.get_gpu_pointer(); }

    auto get_gpu_memory() const { return _impl.get_gpu_memory(); }

    bool gpu_is_expired() const { return _impl.gpu_is_expired(); }

  private:
    /**
     * Initialize a view using a tensor.
     */
    template <TensorConcept TensorType, typename... Args>
        requires(std::is_same_v<T, typename TensorType::ValueType>)
    auto common_initialization(TensorType &other, Dim<Rank> const &dims, Args &&...args) -> void {
        constexpr size_t OtherRank = TensorType::Rank;

        static_assert(Rank <= OtherRank, "A TensorView must be the same Rank or smaller that the Tensor being viewed.");

        _parent = other.data();

        // set_mutex(other.get_mutex());
        Stride<Rank>      default_strides{};
        Offset<OtherRank> default_offsets{};
        Offset<OtherRank> temp_offsets{};
        Stride<Rank>      error_strides{};
        error_strides[0]   = -1;
        Dim<Rank>    _dims = dims;
        Stride<Rank> _strides{};
        T           *_data;

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we compute this
        // dimensionality for them.
        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(dims)) {
            if (dim == -1) {
                nfound++;
                location = i;
            }
        }

        if (nfound > 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "More than one -1 was provided.");
        }

        if (nfound == 1 && Rank == 1) {
            default_offsets.fill(0);
            default_strides.fill(1);

            auto offsets = arguments::get(default_offsets, args...);
            auto strides = arguments::get(default_strides, args...);

            // Perform this with integer arithmetic. There is a chance that with
            // sizes greater than 2^52 that the division becomes inaccurate with
            // floating points. Integer divisions should never become inaccurate.
            // In floating point, it would be ceil((size - offset) / stride)
            ptrdiff_t numerator   = other.size() - offsets[0];
            size_t    denominator = strides[0];

            _dims[location] = numerator / denominator;

            if (numerator % denominator != 0) {
                _dims[location] += 1;
            }
        }

        if (nfound == 1 && Rank > 1) {
            EINSUMS_THROW_EXCEPTION(todo_error, "Haven't coded up this case yet.");
        }

        // If the Ranks are the same then use "other"s stride information
        if constexpr (Rank == OtherRank) {
            default_strides = other.strides();
            // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
            // provide the information
        } else {
            if (std::accumulate(_dims.begin(), _dims.end(), 1.0, std::multiplies()) ==
                std::accumulate(other.dims().begin(), other.dims().end(), 1.0, std::multiplies())) {
                dims_to_strides(_dims, default_strides, other.impl().is_row_major());
            } else {
                // Stride information cannot be automatically deduced.  It must be provided.
                default_strides = arguments::get(error_strides, args...);
                if (default_strides[0] == static_cast<size_t>(-1)) {
                    EINSUMS_THROW_EXCEPTION(bad_logic, "Unable to automatically deduce stride information. Stride must be passed in.");
                }
            }
        }

        default_offsets.fill(0);

        // Use default_* unless the caller provides one to use.
        _strides       = arguments::get(default_strides, args...);
        temp_offsets   = arguments::get(default_offsets, args...);
        size_t ordinal = indices_to_sentinel(other.strides(), temp_offsets);

        // Find source dimensions
        if constexpr (IsIncoreBasicTensorV<TensorType> && IsSameUnderlyingAndRankV<TensorType, TensorView<T, Rank>>) {
            _source_dims    = other.dims();
            _offsets        = temp_offsets;
            _offset_ordinal = ordinal;
            ordinal         = 0;
        } else {
            // In different Ranks, source dimensions can be deduced from the strides.
            // In decreasing strides when matching the correct dimension is reached (dim[i] = 1 is neglected)
            // Where dimenions less than the outermost are skipped these are incorporated into the dimension inner to it
            // If there are no further inner strides, the inner stride is non-zero to account for missing dimensionality.

            if (other.impl().is_column_major()) {
                size_t current_stride    = 1;
                size_t tensor_index      = 0;
                size_t cumulative_stride = 1;
                for (int i = 0; i < Rank; i++) {
                    _source_dims[i]   = 0;
                    cumulative_stride = 1;
                    current_stride    = _strides[i];
                    while (_source_dims[i] == 0) {
                        if (other.stride(tensor_index) == current_stride) {
                            _source_dims[i] = cumulative_stride;
                            _offsets[i]     = temp_offsets[tensor_index];
                        }
                        cumulative_stride *= other.dim(tensor_index);
                        tensor_index++;
                    }
                    if (_source_dims[i] == 0) {
                        EINSUMS_THROW_EXCEPTION(bad_logic,
                                                "Unable to deduce source dimensions. Stride does not follow source tensor dimensions.");
                    }
                }
            } else {
                size_t current_stride    = 1;
                size_t tensor_index      = 0;
                size_t cumulative_stride = 1;
                for (int i = 0; i < Rank; i++) {
                    _source_dims[i]   = 0;
                    cumulative_stride = 1;
                    current_stride    = _strides[i];
                    while (_source_dims[i] == 0) {
                        cumulative_stride *= other.dim(tensor_index);
                        if (other.stride(tensor_index) == current_stride) {
                            _source_dims[i] = cumulative_stride;
                            _offsets[i]     = temp_offsets[tensor_index];
                        }
                        tensor_index++;
                    }
                    if (_source_dims[i] == 0) {
                        EINSUMS_THROW_EXCEPTION(bad_logic,
                                                "Unable to deduce source dimensions. Stride does not follow source tensor dimensions.");
                    }
                }
            }

            _offset_ordinal = indices_to_sentinel(_strides, _offsets); // Only counts offsets that are in the view
            ordinal -= _offset_ordinal;
        }

        if (Rank > 0 && _strides[Rank - 1] == 0) {
            _strides[Rank - 1] = 1;
        }

        for (ptrdiff_t i = (ptrdiff_t)Rank - 2; i >= 0; i--) {
            if (_strides[i] == 0) {
                _strides[i] = _strides[i + 1];
            }
        }

        // Determine the ordinal using the offsets provided (if any) and the strides of the parent
        _data = &(other.data()[_offset_ordinal]);

        _impl = detail::TensorImpl<T>(_data, _dims, _strides);

        for (int i = 0; i < Rank; i++) {
            _dim_array[i]    = _impl.dim(i);
            _stride_array[i] = _impl.stride(i);
        }
    }

    /**
     * @var _name
     *
     * The name of the view used for printing.
     */
    std::string _name{"(Unnamed View)"};

    /**
     * @var _impl
     *
     * The underlying implementation.
     */
    detail::TensorImpl<T> _impl;

    /**
     * @var _source_dims
     *
     * The dimensions of the source tensor.
     */
    Dim<Rank> _source_dims;

    /**
     * @var _offsets
     *
     * These are offsets used to access data from a midpoint in the tensor.
     */
    /**
     * @var _offset_ordinal
     *
     * This is the value at which the offset is currently set. Found by multiplying the offsets by the strides.
     */
    Offset<Rank> _offsets;
    size_t       _offset_ordinal{0};
    Dim<Rank>    _dim_array;
    Stride<Rank> _stride_array;

    T *_parent{nullptr};

    template <typename T_, size_t Rank_, typename Alloc>
    friend struct GeneralTensor;

    template <typename T_, size_t OtherRank_>
    friend struct TensorView;
};

/**
 * Function that zeros a tensor.
 */
template <TensorConcept TensorType>
void zero(TensorType &A) {
    A.zero();
}

#ifndef DOXYGEN
#    ifdef __cpp_deduction_guides
template <typename... Args>
GeneralTensor(std::string const &, Args...) -> GeneralTensor<double, sizeof...(Args), std::allocator<double>>;

template <typename... Args>
GeneralTensor(bool, std::string const &, Args...) -> GeneralTensor<double, sizeof...(Args), std::allocator<double>>;

template <typename T, size_t OtherRank, typename Alloc, typename... Dims>
explicit GeneralTensor(GeneralTensor<T, OtherRank, Alloc> &&otherTensor, std::string name, Dims... dims)
    -> GeneralTensor<T, sizeof...(dims), Alloc>;

template <size_t Rank, typename... Args>
explicit GeneralTensor(Dim<Rank> const &, Args...) -> GeneralTensor<double, Rank, std::allocator<double>>;

template <size_t Rank, typename... Args>
explicit GeneralTensor(bool, Dim<Rank> const &, Args...) -> GeneralTensor<double, Rank, std::allocator<double>>;

template <typename T, size_t Rank, typename Alloc>
GeneralTensor(GeneralTensor<T, Rank, Alloc> const &) -> GeneralTensor<T, Rank, Alloc>;

template <typename T, size_t Rank, typename Alloc>
GeneralTensor(GeneralTensor<T, Rank, Alloc> &&) -> GeneralTensor<T, Rank, Alloc>;

template <typename T, size_t Rank, size_t OtherRank, typename Alloc, typename... Args>
TensorView(GeneralTensor<T, OtherRank, Alloc> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename Alloc, typename... Args>
TensorView(GeneralTensor<T, OtherRank, Alloc> const &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> const &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename Alloc, typename... Args>
TensorView(std::string, GeneralTensor<T, OtherRank, Alloc> &, Dim<Rank> const &, Args...) -> TensorView<T, Rank>;

// Supposedly C++20 will allow template deduction guides for template aliases. i.e. Dim, Stride, Offset, Count, Range.
// Clang has no support for class template argument deduction for alias templates. P1814R0
#    endif
#endif

// Useful factories

/**
 * @brief Create a new tensor with \p name and \p args .
 *
 * Just a simple factory function for creating new tensors. Defaults to using double for the
 * underlying data and automatically determines rank of the tensor from args.
 *
 * A \p name for the tensor is required. \p name is used when printing and performing disk I/O.
 *
 * By default, the allocated tensor data is not initialized to zero. This was a performance
 * decision. In many cases the next step after creating a tensor is to load or store data into
 * it...why waste the CPU cycles zeroing something that will immediately get set to something
 * else.  If you wish to explicitly zero the contents of your tensor use the zero function.
 *
 * @code
 * auto a = create_tensor("a", 3, 3);           // auto -> Tensor<double, 2>
 * auto b = create_tensor<float>("b", 4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam Type The datatype of the underlying tensor. Defaults to double.
 * @tparam Args The datatype of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param name The name of the new tensor.
 * @param args The arguments needed to construct the tensor.
 * @return A new tensor. By default, memory is not initialized to anything. It may be filled with garbage.
 */
template <typename Type = double, bool RowMajor = einsums::row_major_default, typename... Args>
auto create_tensor(std::string const &name, Args... args) {
    EINSUMS_LOG_TRACE("creating tensor {}, {}", name, std::forward_as_tuple(args...));
    return Tensor<Type, sizeof...(Args)>{RowMajor, name, args...};
}

/**
 * @brief Create a new tensor with \p name and \p args .
 *
 * Just a simple factory function for creating new tensors. Defaults to using double for the
 * underlying data and automatically determines rank of the tensor from args.
 *
 * A \p name for the tensor is required. \p name is used when printing and performing disk I/O.
 *
 * By default, the allocated tensor data is not initialized to zero. This was a performance
 * decision. In many cases the next step after creating a tensor is to load or store data into
 * it...why waste the CPU cycles zeroing something that will immediately get set to something
 * else.  If you wish to explicitly zero the contents of your tensor use the zero function.
 *
 * @code
 * auto a = create_tensor(3, 3);           // auto -> Tensor<double, 2>
 * auto b = create_tensor<float>(4, 5, 6); // auto -> Tensor<float, 3>
 * @endcode
 *
 * @tparam Type The datatype of the underlying tensor. Defaults to double.
 * @tparam Args The datatype of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param args The arguments needed to construct the tensor.
 * @return A new tensor. By default, memory is not initialized to anything. It may be filled with garbage.
 */
template <typename Type = double, bool RowMajor = einsums::row_major_default, std::integral... Args>
auto create_tensor(Args... args) {
    return Tensor<Type, sizeof...(Args)>{RowMajor, "Temporary", args...};
}

#ifndef DOXYGEN
template <FileOrOStream Output, RankTensorConcept AType>
    requires(einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>)
void fprintln(Output &fp, AType const &A, TensorPrintOptions options) {
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent indent;
        if constexpr (!TensorViewConcept<AType>)
            fprintln(fp, "Type: In Core Tensor");
        else
            fprintln(fp, "Type: In Core Tensor View");
    }

    if constexpr (AType::Rank == 0) {
        {
            print::Indent const indent{};

            fprintln(fp, "Data Type: {}", type_name<typename AType::ValueType>());

            if (options.full_output) {
                fprintln(fp);

                typename AType::ValueType value = A;

                std::ostringstream oss;
                oss << "              ";
                if constexpr (std::is_floating_point_v<typename AType::ValueType>) {
                    if (std::abs(value) < 1.0E-4) {
                        oss << fmt::format("{:14.4e} ", value);
                    } else {
                        oss << fmt::format("{:14.8f} ", value);
                    }
                } else if constexpr (IsComplexV<typename AType::ValueType>) {
                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                } else
                    oss << fmt::format("{:14} ", value);

                fprintln(fp, "{}", oss.str());
                fprintln(fp);
            }
        }
        fprintln(fp);
    } else {
        fprintln(fp, A.impl(), options);
    }
}

template <RankTensorConcept AType>
    requires(BasicTensorConcept<AType> || !AlgebraTensorConcept<AType>)
void println(AType const &A, TensorPrintOptions options) {
    fprintln(std::cout, A, options);
}

TENSOR_EXPORT_ALLOC_RANK(GeneralTensor, 0, std::allocator)
TENSOR_ALLOC_EXPORT(GeneralTensor, std::allocator)
TENSOR_ALLOC_EXPORT(GeneralTensor, BufferAllocator)

TENSOR_EXPORT(TensorView)
#endif

} // namespace einsums
