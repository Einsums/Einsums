//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#include "Einsums/Config/Types.hpp"
#include "Einsums/TensorImpl/TensorImpl.hpp"
#include "Einsums/TensorImpl/TensorImplOperations.hpp"

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <fmt/format.h>

#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace einsums {

/**
 * @class GeneralRuntimeTensor
 *
 * @brief Represents a tensor whose properties can be determined at runtime but not compile time.
 *
 * This kind of tensor is unable to be used in many of the same ways as a tensor with compile-time rank. It is mostly used for communication
 * with the Python interface.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam Alloc The allocator used for the internal data.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This used to be RuntimeTensor. An allocator parameter was added, and RuntimeTensor is now an alias to this with the standard
 *      allocator.
 * @endversion
 */
template <typename T, typename Alloc>
struct GeneralRuntimeTensor : public tensor_base::CoreTensor,
                              tensor_base::RuntimeTensorNoType,
                              design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef Vector
     *
     * @brief Represents how the data is stored in the tensor.
     */
    using Vector = std::vector<std::remove_cv_t<T>, Alloc>;

    /**
     * @typedef ValueType
     *
     * @brief Represents the data type stored in the tensor.
     */
    using ValueType = T;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = typename detail::TensorImpl<T>::pointer;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = typename detail::TensorImpl<T>::const_pointer;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = typename detail::TensorImpl<T>::reference;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = typename detail::TensorImpl<T>::const_reference;

    GeneralRuntimeTensor() noexcept = default;

    /**
     * @brief Default copy constructor.
     */
    GeneralRuntimeTensor(GeneralRuntimeTensor<T, Alloc> const &copy) : _impl(copy.impl()), _data(copy.vector_data()) {
        _impl.set_data(_data.data());
    }

    /**
     * @brief Copy with a different allocator.
     */
    template <typename Alloc2>
    GeneralRuntimeTensor(GeneralRuntimeTensor<T, Alloc2> const &copy)
        : _impl(copy.impl()), _data(copy.vector_data().begin(), copy.vector_data().end()) {
        _impl.set_data(_data.data());
    }

    template <Container Dim>
    GeneralRuntimeTensor(std::string name, Dim const &dims)
        : _name{name}, _impl(nullptr, dims, GlobalConfigMap::get_singleton().get_bool("row-major")) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    GeneralRuntimeTensor(std::string name, Dim const &dims, bool row_major) : _name{name}, _impl(nullptr, dims, row_major) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given dimensions.
     *
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    explicit GeneralRuntimeTensor(Dim const &dims, bool row_major = row_major_default) : _impl(nullptr, dims, row_major) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions using an initializer list.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor as an initializer list.
     */
    GeneralRuntimeTensor(std::string name, std::initializer_list<size_t> dims, bool row_major = row_major_default)
        : GeneralRuntimeTensor(name, std::vector<size_t>(dims), row_major) {}

    /**
     * @brief Create a new runtime tensor with the given dimensions using an initializer list.
     *
     * @param dims The dimensions of the tensor as an initializer list.
     */
    explicit GeneralRuntimeTensor(std::initializer_list<size_t> dims, bool row_major = row_major_default)
        : GeneralRuntimeTensor(std::vector<size_t>(dims), row_major) {}

    /**
     * @brief Create a new runtime tensor with the given name and dimensions.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    RuntimeTensor(std::string name, Dim const &dims) : _name{name}, _impl(nullptr, dims, GlobalConfigMap::get_singleton().get_bool("row-major")) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given dimensions.
     *
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    explicit RuntimeTensor(Dim const &dims) : _impl(nullptr, dims, GlobalConfigMap::get_singleton().get_bool("row-major")) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions using an initializer list.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor as an initializer list.
     */
    RuntimeTensor(std::string name, std::initializer_list<size_t> dims)
        : RuntimeTensor(name, std::vector<size_t>(dims), GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    /**
     * @brief Create a new runtime tensor with the given dimensions using an initializer list.
     *
     * @param dims The dimensions of the tensor as an initializer list.
     */
    explicit RuntimeTensor(std::initializer_list<size_t> dims)
        : RuntimeTensor(std::vector<size_t>(dims), GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    /**
     * @brief Copy a tensor into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor to copy.
     */
    template <size_t Rank, typename Alloc2>
    GeneralRuntimeTensor(GeneralTensor<T, Rank, Alloc2> const &copy) : _name{copy.name()}, _impl(nullptr, copy.dims(), copy.strides()) {

        _data.resize(copy.size());

        _impl.set_data(_data.data());

        std::memcpy(_data.data(), copy.data(), copy.size() * sizeof(T));
    }

    template <size_t Rank, typename Alloc2>
    GeneralRuntimeTensor(GeneralTensor<T, Rank, Alloc2> &&copy) noexcept
        : _name{std::move(copy.name())}, _impl{std::move(copy.impl())}, _data{std::move(copy.vector_data())} {
        _impl.set_data(_data.data());
    }

    /**
     * @brief Copy a tensor view into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor view to copy.
     */
    template <size_t Rank>
    GeneralRuntimeTensor(TensorView<T, Rank> const &copy) : _impl(nullptr, copy.dims()) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(copy.impl(), _impl);
    }

    /**
     * @brief Copy a tensor view into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor view to copy.
     */
    GeneralRuntimeTensor(RuntimeTensorView<T> const &copy) : _impl(nullptr, copy.dims()) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(copy.impl(), _impl);
    }

    // HIP clang doesn't like it when this is defaulted.
    virtual ~GeneralRuntimeTensor() {}

    /**
     * @brief Set all of the data in the tensor to zero.
     */
    virtual void zero() { std::memset(_data.data(), 0, _data.size() * sizeof(T)); }

    /**
     * @brief Set all of the data in the tensor to the same value.
     *
     * @param val The value to fill the tensor with.
     */
    virtual void set_all(T val) { std::fill(_data.begin(), _data.end(), val); }

    /**
     * @brief Get the pointer to the stored data.
     */
    Pointer data() noexcept { return _data.data(); }

    /**
     * @copydoc data()
     */
    ConstPointer data() const noexcept { return _data.data(); }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param index A collection of integers to use as the index.
     */
    template <Container Storage>
    Pointer data(Storage const &index) {
        return _impl.data(index);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param index A collection of integers to use as the index.
     */
    template <Container Storage>
    ConstPointer data(Storage const &index) const {
        return _impl.data(index);
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. It will also check to see that
     * the correct number of indices were passed.
     *
     * @param index The index to use for the subscript.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage>)
    Reference operator()(Storage const &index) {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. It will also check to see that
     * the correct number of indices were passed.
     *
     * @param index The index to use for the subscript.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage>)
    ConstReference operator()(Storage const &index) const {
        return _impl.subscript(index);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    Pointer data(Args &&...args) {
        return _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    ConstPointer data(Args &&...args) const {
        return _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    Pointer data(Args const &...args) {
        return _impl.data(args...);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    ConstPointer data(Args const &...args) const {
        return _impl.data(args...);
    }

    Reference operator()() { return *_impl.data(); }

    ConstReference operator()() const { return *_impl.data(); }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. If fewer indices than necessary
     * are passed, it will throw an error. This will hopefully change in the future to allow for
     * the creation of views. It will still throw an error when too many arguments are passed.
     *
     * @param args The index to use for the subscript.
     *
     * @todo std::variant can't handle references. We may be able to make our own, but for right now,
     * this will not be able to handle the wrong number of arguments.
     */
    template <std::integral... Args>
    Reference operator()(Args const &...args) {
        return _impl.subscript(args...);
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. If too few indices are passed,
     * it will create a view.
     *
     * @param args The index to use for the subscript.
     */
    template <std::integral... Args>
    ConstReference operator()(Args const &...args) const {
        return _impl.subscript(args...);
    }

    /**
     * @brief Subscripts into the tensor with ranges.
     *
     * This function creates a view based on the ranges passed in.
     *
     * @param args The indices to use. Can be integers, ranges, or All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> const operator()(Args const &...args) const {
        return RuntimeTensorView<T>(_impl.subscript(args...));
    }

    /**
     * @brief Subscripts into the tensor with ranges.
     *
     * This function creates a view based on the ranges passed in.
     *
     * @param args The indices to use. Can be integers, ranges, or All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> operator()(Args const &...args) {
        return RuntimeTensorView<T>(_impl.subscript(args...));
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <size_t Rank, typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralTensor<T, Rank, Alloc2> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank, typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralTensor<TOther, Rank, Alloc2> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    template <typename TOther, size_t Rank>
    GeneralRuntimeTensor &operator=(TensorView<TOther, Rank> const &other) {
        _impl = detail::TensorImpl<T>(nullptr, other.dims());

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    GeneralRuntimeTensor &operator=(GeneralRuntimeTensor<T, Alloc> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    template <typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralRuntimeTensor<T, Alloc2> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    virtual GeneralRuntimeTensor &operator=(RuntimeTensorView<T> const &other) {
        _impl = detail::TensorImpl<T>(nullptr, other.dims());

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralRuntimeTensor<TOther, Alloc2> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    template <typename TOther>
    GeneralRuntimeTensor &operator=(RuntimeTensorView<TOther> const &other) {
        _impl = detail::TensorImpl<T>(nullptr, other.dims());

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Fill the tensor with the given value.
     *
     * @param value The value to fill the tensor with.
     */
    virtual GeneralRuntimeTensor &operator=(T value) {
        set_all(value);
        return *this;
    }

    template <typename TOther>
    GeneralRuntimeTensor &operator+=(TOther const &b) {
        detail::add_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    GeneralRuntimeTensor &operator-=(TOther const &b) {
        detail::sub_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    GeneralRuntimeTensor &operator*=(TOther const &b) {
        detail::mult_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    GeneralRuntimeTensor &operator/=(TOther const &b) {
        detail::div_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator+=(TOther const &b) {
        detail::add_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator-=(TOther const &b) {
        detail::sub_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator*=(TOther const &b) {
        detail::mult_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator/=(TOther const &b) {
        detail::div_assign(b.impl(), _impl);

        return *this;
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() {
        return TensorView<T, Rank>(_impl);
    }

    template <size_t Rank>
    operator TensorView<T, Rank> const() const {
        return TensorView<T, Rank>(_impl);
    }
    /**
     * @brief Get the length of the tensor along a given axis.
     *
     * @param d The axis to query. Negative values will wrap around.
     */
    virtual size_t dim(int d) const { return _impl.dim(d); }

    /**
     * @brief Get the dimensions of the tensor.
     */
    virtual BufferVector<size_t> dims() const noexcept { return _impl.dims(); }

    /**
     * @brief Return the vector containing the data stored by the tensor.
     */
    virtual Vector const &vector_data() const { return _data; }

    /**
     * @brief Return the vector containing the data stored by the tensor.
     */
    virtual Vector &vector_data() { return _data; }

    /**
     * @brief Get the stride along a given axis.
     *
     * @param d The axis to query. Negative values will wrap around.
     */
    virtual size_t stride(int d) const { return _impl.stride(d); }

    /**
     * @brief Return the strides of the tensor.
     */
    virtual BufferVector<size_t> strides() const noexcept { return _impl.strides(); }

    /**
     * @brief Create a rank-1 view of the tensor.
     */
    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        std::vector<size_t> dim{size()};

        return RuntimeTensorView<T>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     */
    virtual auto size() const -> size_t { return _data.size(); }

    /**
     * @brief Returns whether the tensor sees all of the underlying data.
     *
     * This type of tensor will always see all of its underlying data, so this will always be true.
     */
    virtual bool full_view_of_underlying() const noexcept { return true; }

    /**
     * @brief Get the rank of the tensor.
     */
    virtual size_t rank() const noexcept { return _impl.rank(); }

    /**
     * @brief Set the name of the tensor.
     *
     * @param new_name The new name of the tensor.
     */
    virtual void set_name(std::string const &new_name) { this->_name = new_name; }

    /**
     * @brief Get the name of the tensor.
     */
    virtual std::string const &name() const noexcept { return this->_name; }

    virtual detail::TensorImpl<T> &impl() noexcept { return _impl; }

    virtual detail::TensorImpl<T> const &impl() const noexcept { return _impl; }

    bool is_row_major() const { return _impl.is_row_major(); }

    bool is_column_major() const { return _impl.is_column_major(); }

    RuntimeTensorView<T> transpose_view() { return RuntimeTensorView<T>(_impl.transpose_view()); }

    RuntimeTensorView<T> const transpose_view() const { return RuntimeTensorView<T>(_impl.transpose_view()); }

    RuntimeTensorView<T> to_row_major() { return RuntimeTensorView<T>(_impl.to_row_major()); }

    RuntimeTensorView<T> const to_row_major() const { return RuntimeTensorView<T>(_impl.to_row_major()); }

    RuntimeTensorView<T> to_column_major() { return RuntimeTensorView<T>(_impl.to_column_major()); }

    RuntimeTensorView<T> const to_column_major() const { return RuntimeTensorView<T>(_impl.to_column_major()); }

    template <std::integral... MultiIndex>
    RuntimeTensorView<T> tie_indices(MultiIndex &&...index) {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    RuntimeTensorView<T> const tie_indices(MultiIndex &&...index) const {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
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

  protected:
    Vector _data{};

    std::string _name{"(unnamed)"};

    detail::TensorImpl<T> _impl{};

    template <typename TOther>
    friend class RuntimeTensorView;

    template <typename TOther, typename Alloc2>
    friend class GeneralRuntimeTensor;
};

/**
 * @class RuntimeTensorView
 *
 * @brief Represents a view of a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
struct RuntimeTensorView : public tensor_base::CoreTensor,
                           public tensor_base::RuntimeTensorNoType,
                           public tensor_base::RuntimeTensorViewNoType,
                           public design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef ValueType
     *
     * @brief The data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = typename detail::TensorImpl<T>::pointer;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = typename detail::TensorImpl<T>::const_pointer;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = typename detail::TensorImpl<T>::reference;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = typename detail::TensorImpl<T>::const_reference;

    RuntimeTensorView() = default;

    /**
     * @brief Default copy constructor.
     *
     * @param copy The tensor to copy.
     */
    RuntimeTensorView(RuntimeTensorView<T> const &copy) = default;

    /**
     * @brief Creates a new view based on another view.
     *
     * This view and the other view will share the same data pointer.
     *
     * @param view The tensor to view.
     */
    RuntimeTensorView(RuntimeTensor<T> const &view) : _impl{view.impl()}, _name{view.name()} {}

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    template <Container Dim>
    RuntimeTensorView(RuntimeTensor<T> const &other, Dim const &dims)
        : _impl{const_cast<Pointer>(other.data()), dims, other.impl().is_row_major()} {}

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    template <Container Dim>
    RuntimeTensorView(RuntimeTensorView<T> const &other, Dim const &dims)
        : _impl(const_cast<Pointer>(other.data()), dims, other.impl().is_row_major()) {}

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    template <Container Dim, Container Stride, Container Offset, typename Alloc>
    RuntimeTensorView(GeneralRuntimeTensor<T, Alloc> const &other, Dim const &dims, Stride const &strides, Offset const &offsets)
        : _impl(const_cast<Pointer>(other.data(offsets)), dims, strides) {}

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    template <Container Dim, Container Stride, Container Offset>
    RuntimeTensorView(RuntimeTensorView<T> const &other, Dim const &dims, Stride const &strides, Offset const &offsets)
        : _impl(const_cast<Pointer>(other.data(offsets)), dims, strides) {}

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> const &copy) : _impl(copy.impl()) {}

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank, typename Alloc>
    RuntimeTensorView(GeneralTensor<T, Rank, Alloc> const &copy) : _impl(copy.impl()) {}

    /**
     * @brief Creates a view around an implementation.
     */
    RuntimeTensorView(detail::TensorImpl<T> const &impl) : _impl(impl) {}

    // HIP clang doesn't like it when this is defaulted.
    virtual ~RuntimeTensorView() {}

    /**
     * @brief Set all the entries in the tensor to zero.
     */
    virtual void zero() { detail::copy_to(T{0.0}, _impl); }

    /**
     * @brief Fill the tensor with the specified value.
     *
     * @param val The value to fill the tensor with.
     */
    virtual void set_all(T val) { detail::copy_to(val, _impl); }

    /**
     * @brief Return a pointer to the beginning of the data.
     */
    Pointer data() { return _impl.data(); }

    /**
     * @brief Return a pointer to the beginning of the data.
     */
    ConstPointer data() const { return _impl.data(); }

    /**
     * @brief Return a pointer to the data starting at the given index.
     */
    template <Container Storage>
    Pointer data(Storage const &index) {
        return _impl.data(index);
    }

    /**
     * @brief Return a pointer to the data starting at the given index.
     */
    template <Container Storage>
    ConstPointer data(Storage const &index) const {
        return _impl.data(index);
    }

    Reference operator()() { return *_impl.data(); }

    ConstReference operator()() const { return *_impl.data(); }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage>)
    Reference operator()(Storage const &index) {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage>)
    ConstReference operator()(Storage const &index) const {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(std::is_base_of_v<Range, typename Storage::value_type>)
    RuntimeTensorView<T> operator()(Storage const &index) {
        return RuntimeTensorView<T>(_impl.subscript(index));
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(std::is_base_of_v<Range, typename Storage::value_type>)
    RuntimeTensorView<T> const operator()(Storage const &index) const {
        return RuntimeTensorView<T>(_impl.subscript(index));
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    Pointer data(Args &&...args) {
        _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    ConstPointer data(Args &&...args) const {
        _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    Pointer data(Args const &...args) {
        _impl.data(args...);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    ConstPointer data(Args const &...args) const {
        _impl.data(args...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * If there aren't enough indices, an error will be thrown. This version checks for negative indices and does bounds checking.
     *
     * @param args The indices to use for the subscript.
     */
    template <std::integral... Args>
    Reference operator()(Args const &...args) {
        return _impl.subscript(args...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * If there aren't enough indices, an error will be thrown. This version checks for negative indices and does bounds checking.
     *
     * @param args The indices to use for the subscript.
     */
    template <std::integral... Args>
    ConstReference operator()(Args const &...args) const {
        return _impl.subscript(args...);
    }

    /**
     * @brief Create a view with the given parameters.
     *
     * @param args The indices for the subscript. Can contain Range and All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> const operator()(Args const &...args) const {
        return RuntimeTensorView<T>(_impl.subscript(args...));
    }

    /**
     * @brief Create a view with the given parameters.
     *
     * @param args The indices for the subscript. Can contain Range and All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> operator()(Args const &...args) {
        return _impl.subscript(args...);
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(Tensor<TOther, Rank> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(TensorView<TOther, Rank> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    virtual RuntimeTensorView<T> &operator=(RuntimeTensor<T> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    virtual RuntimeTensorView<T> &operator=(BufferRuntimeTensor<T> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    template <typename Alloc>
    RuntimeTensorView<T> &operator=(GeneralRuntimeTensor<T, Alloc> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    virtual RuntimeTensorView<T> &operator=(RuntimeTensorView<T> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, typename Alloc>
    RuntimeTensorView<T> &operator=(GeneralRuntimeTensor<TOther, Alloc> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther>
    RuntimeTensorView<T> &operator=(RuntimeTensorView<TOther> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Fill the tensor with the given value.
     *
     * @param value The value to fill the tensor with.
     */
    virtual RuntimeTensorView<T> &operator=(T value) {
        set_all(value);
        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator+=(TOther const &b) {
        detail::add_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator-=(TOther const &b) {
        detail::sub_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator*=(TOther const &b) {
        detail::mult_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator/=(TOther const &b) {
        detail::div_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator+=(TOther const &b) {
        detail::add_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator-=(TOther const &b) {
        detail::sub_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator*=(TOther const &b) {
        detail::mult_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator/=(TOther const &b) {
        detail::div_assign(b.impl(), _impl);

        return *this;
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }

        return TensorView<T, Rank>(_impl);
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() const {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }

        return TensorView<T, Rank>(_impl);
    }

    template <size_t Rank, typename Alloc>
    operator GeneralTensor<T, Rank, Alloc>() const {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }

        return TensorView<T, Rank>(_impl);
    }

    /**
     * @brief Get the length of the tensor along the given axis.
     *
     * @param d The axis to query. Negative indices will be wrapped around.
     */
    virtual auto dim(int d) const -> size_t { return _impl.dim(d); }

    /**
     * @brief Gets the dimensions of the tensor.
     */
    virtual auto dims() const noexcept -> BufferVector<size_t> { return _impl.dims(); }

    /**
     * @brief Gets the stride of the tensor along the given axis.
     *
     * @param d The axis to query. Negative indices will be wrapped around.
     */
    virtual auto stride(int d) const -> size_t { return _impl.stride(d); }

    /**
     * @brief Gets the strides of the tensor.
     */
    virtual auto strides() const noexcept -> BufferVector<size_t> { return _impl.strides(); }

    /**
     * @brief Gets the rank-1 veiw of the tensor.
     *
     * This does not work well for tensor views due to the variation in strides.
     */
    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        std::vector<size_t> dim{_impl.size()};

        return RuntimeTensorView<T>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     */
    virtual auto size() const noexcept -> size_t { return _impl.size(); }

    /**
     * @brief Checks whether the tensor sees all of the underlying data.
     */
    virtual bool full_view_of_underlying() const noexcept { return _impl.is_contiguous(); }

    /**
     * @brief Returns the name of the tensor.
     */
    virtual std::string const &name() const { return _name; };

    /**
     * @brief Sets the name of the tensor.
     *
     * @param new_name The new name for the tensor.
     */
    virtual void set_name(std::string const &new_name) { _name = new_name; };

    /**
     * @brief Gets the rank of the tensor.
     */
    virtual size_t rank() const noexcept { return _impl.rank(); }

    /**
     * @brief Gets the implementation details.
     */
    virtual detail::TensorImpl<T> &impl() { return _impl; }

    virtual detail::TensorImpl<T> const &impl() const { return _impl; }

    bool is_row_major() const { return _impl.is_row_major(); }

    bool is_column_major() const { return _impl.is_column_major(); }

    RuntimeTensorView<T> transpose_view() { return RuntimeTensorView<T>(_impl.transpose_view()); }

    RuntimeTensorView<T> const transpose_view() const { return RuntimeTensorView<T>(_impl.transpose_view()); }

    RuntimeTensorView<T> to_row_major() { return RuntimeTensorView<T>(_impl.to_row_major()); }

    RuntimeTensorView<T> const to_row_major() const { return RuntimeTensorView<T>(_impl.to_row_major()); }

    RuntimeTensorView<T> to_column_major() { return RuntimeTensorView<T>(_impl.to_column_major()); }

    RuntimeTensorView<T> const to_column_major() const { return RuntimeTensorView<T>(_impl.to_column_major()); }

    template <std::integral... MultiIndex>
    RuntimeTensorView<T> tie_indices(MultiIndex &&...index) {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    RuntimeTensorView<T> const tie_indices(MultiIndex &&...index) const {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
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

  protected:
    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed view)"};

    detail::TensorImpl<T> _impl{};
};

#ifndef DOXYGEN
template <einsums::FileOrOStream Output, einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void fprintln(Output &fp, AType const &A, einsums::TensorPrintOptions options = {}) {
    using namespace einsums;
    using T          = typename AType::ValueType;
    std::size_t Rank = A.rank();

    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};

        if constexpr (!TensorViewConcept<AType>)
            fprintln(fp, "Type: In Core Tensor");
        else
            fprintln(fp, "Type: In Core Tensor View");
    }
    fprintln(fp, A.impl(), options);
}

template <einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void println(AType const &A, einsums::TensorPrintOptions options = {}) {
    fprintln(std::cout, A, options);
}

#endif

#if !defined(EINSUMS_WINDOWS) && !defined(DOXYGEN)
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<float, std::allocator<float>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<double, std::allocator<double>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>;

extern template class EINSUMS_EXPORT GeneralRuntimeTensor<float, BufferAllocator<float>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<double, BufferAllocator<double>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<float>, BufferAllocator<std::complex<float>>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<double>, BufferAllocator<std::complex<double>>>;

extern template class EINSUMS_EXPORT RuntimeTensorView<float>;
extern template class EINSUMS_EXPORT RuntimeTensorView<double>;
extern template class EINSUMS_EXPORT RuntimeTensorView<std::complex<float>>;
extern template class EINSUMS_EXPORT RuntimeTensorView<std::complex<double>>;
#endif
} // namespace einsums