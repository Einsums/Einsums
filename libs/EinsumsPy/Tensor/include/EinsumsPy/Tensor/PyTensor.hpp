//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <memory>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace einsums::python {

// Forward declarations
#ifndef DOXYGEN
template <typename T>
class PyTensor;
#endif

/**
 * @typedef SharedRuntimeTensor
 *
 * @brief Shared pointer to a RuntimeTensor.
 */
template <typename T>
using SharedRuntimeTensor = std::shared_ptr<RuntimeTensor<T>>;

#ifndef DOXYGEN
template <typename T>
class PyTensorView;
#endif

/**
 * @typedef SharedRuntimeTensorView
 *
 * @brief Shared pointer to a RuntimeTensorView.
 */
template <typename T>
using SharedRuntimeTensorView = std::shared_ptr<RuntimeTensorView<T>>;

/**
 * @class PyTensorIterator
 *
 * @brief Walks through the elements of a tensor.
 *
 * @tparam T The type stored in the tensor.
 */
template <typename T>
class EINSUMS_EXPORT PyTensorIterator {
  private:
    /**
     * @property _lock
     *
     * @brief Enhances thread safety so that multiple threads can iterate over a tensor all at once.
     */
    std::mutex mutable _lock;

    /**
     * @property _curr_index
     *
     * @brief Holds where the iterator is currently pointing.
     */
    /**
     * @property _elements
     *
     * @brief Holds the number of elements this iterator will need to cycle through.
     */
    size_t _curr_index, _elements;

    /**
     * @property _index_strides
     *
     * @brief Holds information to be able to turn _curr_index into a list of indices that can be passed to the underlying tensor.
     */
    std::vector<size_t> _index_strides;

    /**
     * @property _tensor
     *
     * @brief The tensor this iterator will iterate over.
     */
    RuntimeTensorView<T> _tensor;

    /**
     * @property _stop
     *
     * @brief Whether the iterator is finished or should keep going.
     */
    /**
     * @property _reverse
     *
     * @brief Indicates whether the iterator is a forward iterator or a reverse iterator.
     */
    bool _stop{false}, _reverse{false};

  public:
    /**
     * @brief Copy constructor with optional direction modification.
     *
     * @param copy The iterator to copy.
     * @param reverse Whether the new iterator is the reverse of the other iterator.
     */
    PyTensorIterator(PyTensorIterator const &copy, bool reverse = false)
        : _curr_index{copy._curr_index}, _elements{copy._elements},
          _index_strides(copy._index_strides.cbegin(), copy._index_strides.cend()), _tensor(copy._tensor),
          _stop{copy._curr_index < 0 || copy._curr_index >= copy._elements}, _reverse{reverse != copy._reverse} {}

    /**
     * Create an iterator around a tensor. Can be reversed.
     *
     * @param other The tensor to walk through.
     * @param reverse Whether to go forward or backward.
     */
    PyTensorIterator(RuntimeTensor<T> const &other, bool reverse = false)
        : _tensor{other}, _reverse{reverse}, _index_strides(other.rank()), _elements(other.size()) {
        if (!reverse) {
            _curr_index = 0;
        } else {
            _curr_index = other.size() - 1;
        }

        dims_to_strides(other.dims(), _index_strides);
    }

    /**
     * Create an iterator around a tensor view. Can be reversed.
     *
     * @param other The tensor view to walk through.
     * @param reverse Whether to go forward or backward.
     */
    PyTensorIterator(RuntimeTensorView<T> const &other, bool reverse = false)
        : _tensor{other}, _reverse{reverse}, _index_strides(other.rank()), _elements(other.size()) {
        if (!reverse) {
            _curr_index = 0;
        } else {
            _curr_index = other.size() - 1;
        }

        dims_to_strides(other.dims(), _index_strides);
    }

    /**
     * Get the next element in the tensor.
     */
    T next() {
        auto guard = std::lock_guard(_lock);

        if (_stop) {
            throw pybind11::stop_iteration();
        }

        std::vector<size_t> ind(_tensor.rank());

        sentinel_to_indices(_curr_index, _index_strides, ind);

        T &out = _tensor(ind);

        if (reversed()) {
            _curr_index--;
        } else {
            _curr_index++;
        }

        if (_curr_index < 0 || _curr_index >= _elements) {
            _stop = true;
        }

        return out;
    }

    /**
     * Returns whether the iterator is stepping forward or backward.
     */
    bool reversed() const noexcept { return _reverse; }
};

/**
 * @class PyTensor
 *
 * @brief A Python wrapper for RuntimeTensor.
 *
 * @tparam The type stored by the tensor.
 */
template <typename T>
class PyTensor : public RuntimeTensor<T> {
  public:
    using RuntimeTensor<T>::RuntimeTensor;

    /**
     * @brief Copy constructor from shared pointer.
     */
    PyTensor(std::shared_ptr<PyTensor<T>> const &other) : PyTensor(*other) {}

    /**
     * Create a tensor from a Python buffer object.
     */
    PyTensor(pybind11::buffer const &buffer) {
        auto buffer_info = buffer.request();

        this->_rank = buffer_info.ndim;
        this->_dims.resize(this->_rank);
        this->_strides.resize(this->_rank);

        size_t new_size = 1;
        bool   is_view  = false;
        for (int i = buffer_info.ndim - 1; i >= 0; i--) {
            this->_dims[i]    = buffer_info.shape[i];
            this->_strides[i] = new_size;
            new_size *= this->_dims[i];

            if (this->_strides[i] != buffer_info.strides[i] / buffer_info.itemsize) {
                is_view = true;
            }
        }

        this->_data.resize(new_size);

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            T *buffer_data = (T *)buffer_info.ptr;
            if (is_view) {
                EINSUMS_OMP_PARALLEL_FOR
                for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {
                    size_t buffer_sent = 0, hold = sentinel;
                    for (int i = 0; i < this->_rank; i++) {
                        buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_strides[i]);
                        hold %= this->_strides[i];
                    }
                    this->_data[sentinel] = buffer_data[buffer_sent];
                }
            } else {
                std::memcpy(this->_data.data(), buffer_data, sizeof(T) * this->_data.size());
            }
        } else {
            copy_and_cast_assign(buffer_info, is_view);
        }
    }

    virtual ~PyTensor() = default;

    /**
     * Clear the tensor.
     */
    void zero() override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, zero); }

    /**
     * Set the tensor to the given values.
     *
     * @param val The value to set the tensor to.
     */
    void set_all(T val) override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, set_all, val); }

  private:
    /**
     * @brief Subscript the tensor to get a value.
     *
     * @param args The index of the value.
     */
    T &subscript_to_val(pybind11::tuple const &args) {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a value.
     *
     * @param args The index of the value.
     */
    T const &subscript_to_val(pybind11::tuple const &args) const {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a view
     *
     * @param args The index of the view. Can contain slices.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::tuple const &args) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a view
     *
     * @param args The index of the view. Can contain slices.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::tuple const &args) const {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    /**
     * @brief Set the value of a certain position.
     *
     * @param value The value to set.
     * @param index The index of the position.
     */
    void set_value_at(T value, std::vector<ptrdiff_t> const &index) {
        T &target = this->operator()(index);
        target    = value;
        return target;
    }

    /**
     * @brief Copy multiple values to multiple positions.
     *
     * @param view The values to copy.
     * @param args The indices to copy to. Can contain slices.
     */
    void assign_to_view(pybind11::buffer const &view, pybind11::tuple const &args) {
        PyTensorView<T> this_view = subscript_to_view(args);

        this_view = view;
    }

    /**
     * @brief Copy one value to multiple positions.
     *
     * @param value The value to set.
     * @param args The indices to set. Can contain slices.
     */
    void assign_to_view(T value, pybind11::tuple const &args) {
        auto this_view = subscript_to_view(args);

        this_view = value;
    }

  public:
    /**
     * @brief Subscript into the tensor. Can contain slices.
     *
     * @param args The indices to use to subscript.
     * @return A Python object containing a reference to a single value or a RuntimeTensorView.
     */
    pybind11::object subscript(pybind11::tuple const &args) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__subscript");

        if (override) {
            auto o = override(args);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (args.size() < this->_rank) {
                return pybind11::cast(subscript_to_view(args));
            }
            if (args.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < args.size(); i++) {
                auto const &arg = args[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    return pybind11::cast(subscript_to_view(args));
                }
            }

            return pybind11::cast(subscript_to_val(args));
        }
    }

    /**
     * @brief Copy multiple values into a section of the tensor.
     *
     * @param value The values to copy.
     * @param index The indices to use to subscript.
     * @return A Python object containing a reference to a single value or a RuntimeTensorView.
     */
    pybind11::object assign_values(pybind11::buffer const &value, pybind11::tuple const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(subscript_to_view(index));
            }
            if (index.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                auto const &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    return subscript(index);
                }
            }
            EINSUMS_THROW_EXCEPTION(std::length_error, "Can not assign buffer object to a single position!");
        }
    }

    /**
     * @brief Copy a single value over a section of the tensor.
     *
     * @param value The value to copy.
     * @param index The indices to use to subscript.
     * @return A Python object containing a reference to a single value or a RuntimeTensorView.
     */
    pybind11::object assign_values(T value, pybind11::tuple const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(subscript_to_view(index));
            }
            if (index.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                auto const &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    return subscript(index);
                }
            }

            T &target = subscript_to_val(index);
            target    = value;

            return subscript(index);
        }
    }

    /**
     * @brief Subscript into the tensor using a single slice.
     *
     * @param arg The slice to use for the tensor.
     * @return The view containing the slice.
     */
    RuntimeTensorView<T> subscript(pybind11::slice const &arg) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__subscript");

        if (override) {
            auto o = override(arg);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(arg).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass);
        }
    }

    /**
     * @brief Assign multiple values from a buffer into a slice of the tensor.
     *
     * @param value The values to copy.
     * @param index The slice to use for the tensor.
     * @return The view containing the slice.
     */
    RuntimeTensorView<T> assign_values(pybind11::buffer const &value, pybind11::slice const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass) = value;
        }
    }

    /**
     * @brief Assign a single value over a slice of the tensor.
     *
     * @param value The value to copy.
     * @param index The slice to use for the tensor.
     * @return The view containing the slice.
     */
    RuntimeTensorView<T> assign_values(T value, pybind11::slice const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass) = value;
        }
    }

    /**
     * @brief Get the value at a certain index.
     *
     * @param index The index to use.
     * @return A Python object containing either a single value if the tensor is rank-1 or a RuntimeTensorView otherwise.
     */
    pybind11::object subscript(ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__subscript");

        if (override) {
            auto o = override(index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (this->_rank == 1) {
                return pybind11::cast(this->operator()(index));
            } else {
                return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
            }
        }
    }

    /**
     * @brief Copy values into a view of the tensor whose first index is set.
     *
     * @param value The buffer to copy.
     * @param index The index to use.
     * @return The view containing the the modified indices.
     */
    RuntimeTensorView<T> assign_values(pybind11::buffer const &value, ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            if (this->_rank <= 1) {
                EINSUMS_THROW_EXCEPTION(std::length_error, "Can not assign buffer to a single position!");
            }

            return this->operator()(std::vector<Range>{Range{-1, index}}) = value;
        }
    }

    /**
     * @brief Copy a value over a section of a tensor.
     *
     * @param value The value to copy.
     * @param index The index to use.
     * @return A Python object containing either a single value if the tensor is rank-1 or a RuntimeTensorView otherwise.
     */
    pybind11::object assign_values(T value, ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (this->_rank <= 1) {
                T &target = this->operator()({index});
                target    = value;
                return pybind11::cast(target);
            }

            auto view = this->operator()(std::vector<Range>{Range{-1, index}});
            view      = value;
            return pybind11::cast(view);
        }
    }

    /**
     * @brief Assign a buffer to this tensor, reshaping if necessary.
     *
     * @param buffer The buffer to assign.
     * @return A reference to this.
     */
    RuntimeTensor<T> &operator=(pybind11::buffer const &buffer) {
        auto buffer_info = buffer.request();

        if (this->rank() != buffer_info.ndim) {
            this->_rank = buffer_info.ndim;
            this->_dims.resize(this->_rank);
            this->_strides.resize(this->_rank);
        }

        size_t new_size = 1;
        bool   is_view  = false;
        for (int i = buffer_info.ndim - 1; i >= 0; i--) {
            this->_dims[i]    = buffer_info.shape[i];
            this->_strides[i] = new_size;
            new_size *= this->_dims[i];

            if (this->_strides[i] != buffer_info.strides[i] / buffer_info.itemsize) {
                is_view = true;
            }
        }

        if (new_size != this->_data.size()) {
            this->_data.resize(new_size);
        }

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            T *buffer_data = (T *)buffer_info.ptr;
            if (is_view) {
                EINSUMS_OMP_PARALLEL_FOR
                for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {
                    size_t buffer_sent = 0, hold = sentinel;
                    for (int i = 0; i < this->_rank; i++) {
                        buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_strides[i]);
                        hold %= this->_strides[i];
                    }
                    this->_data[sentinel] = buffer_data[buffer_sent];
                }
            } else {
                std::memcpy(this->_data.data(), buffer_data, sizeof(T) * this->_data.size());
            }
        } else {
            copy_and_cast_assign(buffer_info, is_view);
        }
        return *this;
    }

  private:
#define COPY_CAST_OP(OP, NAME)                                                                                                             \
    /**                                                                                                                                    \
     * @brief Copy values from a buffer into this, casting as necessary. Can also perform in-place operations.                             \
     *                                                                                                                                     \
     * @param buffer_info The buffer to operate on.                                                                                        \
     * @param is_view Whether the buffer is a view, which would necessitate involving strides.                                             \
     */                                                                                                                                    \
    template <typename TOther>                                                                                                             \
    void copy_and_cast_imp_##NAME(const pybind11::buffer_info &buffer_info, bool is_view) {                                                \
        TOther *buffer_data = (TOther *)buffer_info.ptr;                                                                                   \
        if (is_view) {                                                                                                                     \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {                                                               \
                size_t buffer_sent = 0, hold = sentinel;                                                                                   \
                for (int i = 0; i < this->_rank; i++) {                                                                                    \
                    buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_strides[i]);                           \
                    hold %= this->_strides[i];                                                                                             \
                }                                                                                                                          \
                if constexpr (std::is_base_of_v<pybind11::object, TOther>) {                                                               \
                    this->_data[sentinel] OP pybind11::cast<T>((TOther)buffer_data[buffer_sent]);                                          \
                } else if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                 \
                    this->_data[sentinel] OP(T)(RemoveComplexT<T>) buffer_data[buffer_sent];                                               \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    this->_data[sentinel] OP(T) buffer_data[buffer_sent].real();                                                           \
                } else {                                                                                                                   \
                    this->_data[sentinel] OP(T) buffer_data[buffer_sent];                                                                  \
                }                                                                                                                          \
            }                                                                                                                              \
        } else {                                                                                                                           \
            EINSUMS_OMP_PARALLEL_FOR_SIMD                                                                                                  \
            for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {                                                               \
                if constexpr (std::is_base_of_v<pybind11::object, TOther>) {                                                               \
                    this->_data[sentinel] OP pybind11::cast<T>((TOther)buffer_data[sentinel]);                                             \
                } else if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                 \
                    this->_data[sentinel] OP(T)(RemoveComplexT<T>) buffer_data[sentinel];                                                  \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    this->_data[sentinel] OP(T) buffer_data[sentinel].real();                                                              \
                } else {                                                                                                                   \
                    this->_data[sentinel] OP(T) buffer_data[sentinel];                                                                     \
                }                                                                                                                          \
            }                                                                                                                              \
        }                                                                                                                                  \
    }                                                                                                                                      \
    void copy_and_cast_##NAME(const pybind11::buffer_info &buffer_info, bool is_view) {                                                    \
        auto format = buffer_info.format;                                                                                                  \
        if (format.length() > 2) {                                                                                                         \
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can't handle user defined data type {}!", format);                              \
        }                                                                                                                                  \
        switch (format[0]) {                                                                                                               \
        case 'b':                                                                                                                          \
            copy_and_cast_imp_##NAME<int8_t>(buffer_info, is_view);                                                                        \
            break;                                                                                                                         \
        case 'B':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint8_t>(buffer_info, is_view);                                                                       \
            break;                                                                                                                         \
        case 'h':                                                                                                                          \
            copy_and_cast_imp_##NAME<int16_t>(buffer_info, is_view);                                                                       \
            break;                                                                                                                         \
        case 'H':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint16_t>(buffer_info, is_view);                                                                      \
            break;                                                                                                                         \
        case 'i':                                                                                                                          \
            copy_and_cast_imp_##NAME<int32_t>(buffer_info, is_view);                                                                       \
            break;                                                                                                                         \
        case 'I':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint32_t>(buffer_info, is_view);                                                                      \
            break;                                                                                                                         \
        case 'q':                                                                                                                          \
            copy_and_cast_imp_##NAME<int64_t>(buffer_info, is_view);                                                                       \
            break;                                                                                                                         \
        case 'Q':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint64_t>(buffer_info, is_view);                                                                      \
            break;                                                                                                                         \
        case 'l':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                copy_and_cast_imp_##NAME<int32_t>(buffer_info, is_view);                                                                   \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                copy_and_cast_imp_##NAME<int64_t>(buffer_info, is_view);                                                                   \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'L':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                copy_and_cast_imp_##NAME<uint32_t>(buffer_info, is_view);                                                                  \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                copy_and_cast_imp_##NAME<uint64_t>(buffer_info, is_view);                                                                  \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'f':                                                                                                                          \
            copy_and_cast_imp_##NAME<float>(buffer_info, is_view);                                                                         \
            break;                                                                                                                         \
        case 'd':                                                                                                                          \
            copy_and_cast_imp_##NAME<double>(buffer_info, is_view);                                                                        \
            break;                                                                                                                         \
        case 'g':                                                                                                                          \
            copy_and_cast_imp_##NAME<long double>(buffer_info, is_view);                                                                   \
            break;                                                                                                                         \
        case 'Z':                                                                                                                          \
            switch (format[1]) {                                                                                                           \
            case 'f':                                                                                                                      \
                copy_and_cast_imp_##NAME<std::complex<float>>(buffer_info, is_view);                                                       \
                break;                                                                                                                     \
            case 'd':                                                                                                                      \
                copy_and_cast_imp_##NAME<std::complex<double>>(buffer_info, is_view);                                                      \
                break;                                                                                                                     \
            case 'g':                                                                                                                      \
                copy_and_cast_imp_##NAME<std::complex<long double>>(buffer_info, is_view);                                                 \
                break;                                                                                                                     \
            default:                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,                 \
                                        pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                                 \
            }                                                                                                                              \
            break;                                                                                                                         \
        default:                                                                                                                           \
            EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,                     \
                                    pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                                     \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign)
    COPY_CAST_OP(+=, add)
    COPY_CAST_OP(-=, sub)
    COPY_CAST_OP(*=, mult)
    COPY_CAST_OP(/=, div)
#undef COPY_CAST_OP

  public:
#define OPERATOR(OP, NAME, OPNAME)                                                                                                         \
    template <typename TOther>                                                                                                             \
    RuntimeTensor<T> &operator OP(const TOther & other) {                                                                                  \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OPNAME, other);                                                            \
    }                                                                                                                                      \
    template <typename TOther>                                                                                                             \
    RuntimeTensor<T> &operator OP(const RuntimeTensor<TOther> &other) {                                                                    \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OPNAME, other);                                                            \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    RuntimeTensor<T> &operator OP(const RuntimeTensorView<TOther> &other) {                                                                \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OPNAME, other);                                                            \
    }                                                                                                                                      \
    RuntimeTensor<T> &operator OP(const pybind11::buffer & buffer) {                                                                       \
        pybind11::gil_scoped_acquire gil;                                                                                                  \
        pybind11::function           override = pybind11::get_override(static_cast<PyTensor<T> *>(this), #OPNAME);                         \
                                                                                                                                           \
        if (override) {                                                                                                                    \
            auto o = override(buffer);                                                                                                     \
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensor<T> &>::value) {                                          \
                static pybind11::detail::override_caster_t<RuntimeTensor<T> &> caster;                                                     \
                return pybind11::detail::cast_ref<RuntimeTensor<T> &>(std::move(o), caster);                                               \
            }                                                                                                                              \
            return pybind11::detail::cast_safe<RuntimeTensor<T> &>(std::move(o));                                                          \
        } else {                                                                                                                           \
            auto buffer_info = buffer.request();                                                                                           \
                                                                                                                                           \
            if (this->rank() != buffer_info.ndim) {                                                                                        \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform " #OP " with buffer object with different rank!");           \
            }                                                                                                                              \
                                                                                                                                           \
            bool is_view = false;                                                                                                          \
            for (int i = buffer_info.ndim - 1; i >= 0; i--) {                                                                              \
                if (this->_dims[i] != buffer_info.shape[i]) {                                                                              \
                    EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform " #OP " with buffer object with different dimensions!");     \
                }                                                                                                                          \
                                                                                                                                           \
                if (this->_strides[i] != buffer_info.strides[i] / buffer_info.itemsize) {                                                  \
                    is_view = true;                                                                                                        \
                }                                                                                                                          \
            }                                                                                                                              \
                                                                                                                                           \
            if (buffer_info.item_type_is_equivalent_to<T>()) {                                                                             \
                T *buffer_data = (T *)buffer_info.ptr;                                                                                     \
                if (is_view) {                                                                                                             \
                    EINSUMS_OMP_PARALLEL_FOR                                                                                               \
                    for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {                                                       \
                        size_t buffer_sent = 0, hold = sentinel;                                                                           \
                        for (int i = 0; i < this->_rank; i++) {                                                                            \
                            buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_strides[i]);                   \
                            hold %= this->_strides[i];                                                                                     \
                        }                                                                                                                  \
                        this->_data[sentinel] OP buffer_data[buffer_sent];                                                                 \
                    }                                                                                                                      \
                } else {                                                                                                                   \
                    EINSUMS_OMP_PARALLEL_FOR_SIMD                                                                                          \
                    for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {                                                       \
                        this->_data[sentinel] OP buffer_data[sentinel];                                                                    \
                    }                                                                                                                      \
                }                                                                                                                          \
            } else {                                                                                                                       \
                copy_and_cast_##NAME(buffer_info, is_view);                                                                                \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
    }

    OPERATOR(*=, mult, operator*=)
    OPERATOR(/=, div, operator/=)
    OPERATOR(+=, add, operator+=)
    OPERATOR(-=, sub, operator-=)

#undef OPERATOR

    /**
     * @brief Get the length of the tensor along a given direction.
     *
     * @param d The dimension to query.
     */
    size_t dim(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, dim, d); }

    /**
     * @brief Get the dimensions of the tensor.
     */
    std::vector<size_t> dims() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensor<T>, dims); }

    /**
     * @brief Get the vector holding the tensor's data.
     */
    typename RuntimeTensor<T>::Vector const &vector_data() const noexcept override {
        PYBIND11_OVERRIDE(typename RuntimeTensor<T>::Vector const &, RuntimeTensor<T>, vector_data);
    }

    /**
     * @brief Get the vector holding the tensor's data.
     */
    typename RuntimeTensor<T>::Vector &vector_data() noexcept override {
        PYBIND11_OVERRIDE(typename RuntimeTensor<T>::Vector &, RuntimeTensor<T>, vector_data);
    }

    /**
     * @brief Get the stride of the tensor along a given axis.
     *
     * @param d The axis to query.
     */
    size_t stride(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, stride, d); }

    /**
     * @brief Get the strides of the tensor.
     */
    std::vector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensor<T>, strides); }

    /**
     * @brief Create a rank-1 view of the tensor.
     */
    RuntimeTensorView<T> to_rank_1_view() const override { PYBIND11_OVERRIDE(RuntimeTensorView<T>, RuntimeTensor<T>, to_rank_1_view); }

    /**
     * @brief Check whether the tensor can see all of the data it stores.
     *
     * This function should return true for this kind of tensor.
     */
    bool full_view_of_underlying() const noexcept override { PYBIND11_OVERRIDE(bool, RuntimeTensor<T>, full_view_of_underlying); }

    /**
     * @brief Get the tensor's name.
     */
    std::string const &name() const noexcept override { PYBIND11_OVERRIDE_NAME(std::string const &, RuntimeTensor<T>, "get_name", name); }

    /**
     * @brief Set the tensor's name.
     *
     * @param new_name The new name for the tensor.
     */
    void set_name(std::string const &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, set_name, new_name); }

    /**
     * @brief Get the rank of the tensor.
     */
    size_t rank() const noexcept override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, rank); }
};

/**
 * @class PyTensorView
 *
 * @brief Views a runtime tensor and makes it available to Python.
 *
 * @see PyTensor for information on methods.
 */
template <typename T>
class PyTensorView : public RuntimeTensorView<T> {
  public:
    using RuntimeTensorView<T>::RuntimeTensorView;

    /**
     * @brief Default copy constructor.
     *
     * This creates a new view that points to the same tensor as the input.
     */
    PyTensorView(PyTensorView<T> const &) = default;

    /**
     * @brief Create a view of the given tensor.
     *
     * @param copy The tensor to view.
     */
    PyTensorView(RuntimeTensorView<T> const &copy) : RuntimeTensorView<T>(copy) {}

    /**
     * @brief Create a view of the given buffer.
     *
     * @param buffer The buffer to view.
     */
    PyTensorView(pybind11::buffer &buffer) {
        pybind11::buffer_info buffer_info = buffer.request(true);

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            this->_data = (T *)buffer_info.ptr;
        } else {
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not create RuntimeTensorView from buffer whose type does not match!");
        }

        this->_rank = buffer_info.ndim;
        this->_dims.resize(this->_rank);
        this->_strides.resize(this->_rank);
        this->_index_strides.resize(this->_rank);
        this->_size       = 1;
        this->_alloc_size = buffer_info.shape[0] * buffer_info.strides[0];

        for (int i = this->_rank - 1; i >= 0; i--) {
            this->_strides[i]       = buffer_info.strides[i] / buffer_info.itemsize;
            this->_dims[i]          = buffer_info.shape[i];
            this->_index_strides[i] = this->_size;
            this->_size *= this->_dims[i];
        }
    }

    virtual ~PyTensorView() = default;

    /**
     * @brief Set all values in the view to zero.
     */
    void zero() override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, zero); }

    /**
     * @brief Fill the view with the given value.
     *
     * @param val The value to fill the tensor with.
     */
    void set_all(T val) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_all, val); }

  private:
    /**
     * @brief Worker method that subscripts into the view and returns a reference to the requested element.
     *
     * @param args The indices to use for the subscript.
     */
    T &subscript_to_val(pybind11::tuple const &args) {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Worker method that subscripts into the view and returns a reference to the requested element.
     *
     * @param args The indices to use for the subscript.
     */
    T const &subscript_to_val(pybind11::tuple const &args) const {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Worker method that creates a view based on the indices and slices passed in.
     *
     * @param args The indices and slices to use for view creation.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::tuple const &args) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    /**
     * @brief Worker method that creates a view based on the indices and slices passed in.
     *
     * @param args The indices and slices to use for view creation.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::tuple const &args) const {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    /**
     * @brief Set the value at the given point in the tensor to the given value.
     *
     * @param value The new value.
     * @param index Where to set the value.
     */
    void set_value_at(T value, std::vector<size_t> const &index) {
        T &target = this->operator()(index);
        target    = value;
        return target;
    }

    /**
     * @brief Copy the data from a buffer into this view.
     *
     * Creates a view of part of the tensor using the subscript arguments, then
     * assigns the buffer to that view.
     *
     * @param view The buffer to copy.
     * @param args The position to copy to.
     */
    void assign_to_view(pybind11::buffer const &view, pybind11::tuple const &args) {
        PyTensorView<T> this_view = subscript_to_view(args);

        this_view = view;
    }

    /**
     * @brief Fill part of the view with the given value.
     *
     * @param value The value to fill the view with.
     * @param args Indices and slices that determine the part of the view to fill.
     */
    void assign_to_view(T value, pybind11::tuple const &args) {
        auto this_view = subscript_to_view(args);

        this_view = value;
    }

  public:
    /**
     * @brief Subscript into the tensor.
     *
     * This method handles view creation when necessary.
     *
     * @param args The indices and slices to use for the view creation.
     */
    pybind11::object subscript(pybind11::tuple const &args) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__subscript");

        if (override) {
            auto o = override(args);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (args.size() < this->_rank) {
                return pybind11::cast(subscript_to_view(args));
            }
            if (args.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < args.size(); i++) {
                auto const &arg = args[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    return pybind11::cast(subscript_to_view(args));
                }
            }

            return pybind11::cast(subscript_to_val(args));
        }
    }

    /**
     * @brief Assign a buffer to part of the view.
     *
     * @param value The buffer to assign from.
     * @param index The indices and slices determining where to assign to.
     */
    pybind11::object assign_values(pybind11::buffer const &value, pybind11::tuple const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(*this);
            }
            if (index.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                auto const &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    return subscript(index);
                }
            }
            EINSUMS_THROW_EXCEPTION(std::length_error, "Can not assign buffer object to a single position!");
        }
    }

    /**
     * @brief Fill part of the view with a value.
     *
     * @param value The value to fill the view with
     * @param index The indices and slices determining where to assign to.
     */
    pybind11::object assign_values(T value, pybind11::tuple const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(*this);
            }
            if (index.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                auto const &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    return subscript(index);
                }
            }

            T &target = subscript_to_val(index);
            target    = value;

            return subscript(index);
        }
    }

    /**
     * @brief Subscript into the tensor using a slice.
     *
     * @param arg The slice to use to subscript.
     */
    RuntimeTensorView<T> subscript(pybind11::slice const &arg) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__subscript");

        if (override) {
            auto o = override(arg);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(arg).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass);
        }
    }

    /**
     * @brief Assign a buffer to part of the view.
     *
     * @param value The buffer to assign from.
     * @param index The slice determining where to assign to.
     */
    RuntimeTensorView<T> assign_values(pybind11::buffer const &value, pybind11::slice const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass) = value;
        }
    }

    /**
     * @brief Fill part of the view with a value.
     *
     * @param value The value to fill the view with
     * @param index The slice determining where to assign to.
     */
    RuntimeTensorView<T> assign_values(T value, pybind11::slice const &index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass) = value;
        }
    }

    /**
     * @brief Subscript into the tensor using a single value.
     *
     * Creates a view if the rank is greater than 1.
     *
     * @param index The index for the tensor.
     */
    pybind11::object subscript(ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__subscript");

        if (override) {
            auto o = override(index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (this->_rank == 1) {
                return pybind11::cast(this->operator()(index));
            } else {
                return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
            }
        }
    }

    /**
     * @brief Assign a buffer to the position specified by the index.
     *
     * @param value The buffer to assign from.
     * @param index The index to use to determine the view to assign to.
     */
    RuntimeTensorView<T> assign_values(pybind11::buffer const &value, ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11::detail::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            if (this->_rank <= 1) {
                EINSUMS_THROW_EXCEPTION(std::length_error, "Can not assign buffer to a single position!");
            }

            return this->operator()(std::vector<Range>{Range{-1, index}}) = value;
        }
    }

    /**
     * @brief Fill part of the view with a value.
     *
     * @param value The value to fill the view.
     * @param index The index to use to determine the view to assign to.
     */
    pybind11::object assign_values(T value, ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (this->_rank <= 1) {
                T &target = this->operator()({index});
                target    = value;
                return pybind11::cast(target);
            }

            auto view = this->operator()(std::vector<Range>{Range{-1, index}});
            view      = value;
            return pybind11::cast(view);
        }
    }

  private:
#define COPY_CAST_OP(OP, NAME)                                                                                                             \
    template <typename TOther>                                                                                                             \
    void copy_and_cast_imp_##NAME(const pybind11::buffer_info &buffer_info) {                                                              \
        TOther *buffer_data = (TOther *)buffer_info.ptr;                                                                                   \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {                                                                   \
            size_t buffer_sent = 0, hold = sentinel, ord = 0;                                                                              \
            for (int i = 0; i < this->_rank; i++) {                                                                                        \
                ord += this->_strides[i] * (hold / this->_index_strides[i]);                                                               \
                buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_index_strides[i]);                         \
                hold %= this->_index_strides[i];                                                                                           \
            }                                                                                                                              \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                this->_data[ord] OP(T)(RemoveComplexT<T>) buffer_data[buffer_sent];                                                        \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                this->_data[ord] OP(T) buffer_data[buffer_sent].real();                                                                    \
            } else {                                                                                                                       \
                this->_data[ord] OP(T) buffer_data[buffer_sent];                                                                           \
            }                                                                                                                              \
        }                                                                                                                                  \
    }                                                                                                                                      \
    void copy_and_cast_##NAME(const pybind11::buffer_info &buffer_info) {                                                                  \
        auto format = buffer_info.format;                                                                                                  \
        if (format.length() > 2) {                                                                                                         \
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can't handle user defined data type {}!", format);                              \
        }                                                                                                                                  \
        switch (format[0]) {                                                                                                               \
        case 'b':                                                                                                                          \
            copy_and_cast_imp_##NAME<int8_t>(buffer_info);                                                                                 \
            break;                                                                                                                         \
        case 'B':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint8_t>(buffer_info);                                                                                \
            break;                                                                                                                         \
        case 'h':                                                                                                                          \
            copy_and_cast_imp_##NAME<int16_t>(buffer_info);                                                                                \
            break;                                                                                                                         \
        case 'H':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint16_t>(buffer_info);                                                                               \
            break;                                                                                                                         \
        case 'i':                                                                                                                          \
            copy_and_cast_imp_##NAME<int32_t>(buffer_info);                                                                                \
            break;                                                                                                                         \
        case 'I':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint32_t>(buffer_info);                                                                               \
            break;                                                                                                                         \
        case 'q':                                                                                                                          \
            copy_and_cast_imp_##NAME<int64_t>(buffer_info);                                                                                \
            break;                                                                                                                         \
        case 'Q':                                                                                                                          \
            copy_and_cast_imp_##NAME<uint64_t>(buffer_info);                                                                               \
            break;                                                                                                                         \
        case 'l':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                copy_and_cast_imp_##NAME<int32_t>(buffer_info);                                                                            \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                copy_and_cast_imp_##NAME<int64_t>(buffer_info);                                                                            \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'L':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                copy_and_cast_imp_##NAME<uint32_t>(buffer_info);                                                                           \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                copy_and_cast_imp_##NAME<uint64_t>(buffer_info);                                                                           \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'f':                                                                                                                          \
            copy_and_cast_imp_##NAME<float>(buffer_info);                                                                                  \
            break;                                                                                                                         \
        case 'd':                                                                                                                          \
            copy_and_cast_imp_##NAME<double>(buffer_info);                                                                                 \
            break;                                                                                                                         \
        case 'g':                                                                                                                          \
            copy_and_cast_imp_##NAME<long double>(buffer_info);                                                                            \
            break;                                                                                                                         \
        case 'Z':                                                                                                                          \
            switch (format[1]) {                                                                                                           \
            case 'f':                                                                                                                      \
                copy_and_cast_imp_##NAME<std::complex<float>>(buffer_info);                                                                \
                break;                                                                                                                     \
            case 'd':                                                                                                                      \
                copy_and_cast_imp_##NAME<std::complex<double>>(buffer_info);                                                               \
                break;                                                                                                                     \
            case 'g':                                                                                                                      \
                copy_and_cast_imp_##NAME<std::complex<long double>>(buffer_info);                                                          \
                break;                                                                                                                     \
            default:                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,                 \
                                        pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                                 \
            }                                                                                                                              \
            break;                                                                                                                         \
        default:                                                                                                                           \
            EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,                     \
                                    pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                                     \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign)
    COPY_CAST_OP(+=, add)
    COPY_CAST_OP(-=, sub)
    COPY_CAST_OP(*=, mult)
    COPY_CAST_OP(/=, div)
#undef COPY_CAST_OP

  public:
    /**
     * @brief Copy the data from a buffer into the view.
     *
     * @param buffer The buffer to copy.
     */
    PyTensorView<T> &operator=(pybind11::buffer const &buffer) {
        auto buffer_info = buffer.request();

        if (this->rank() != buffer_info.ndim) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not change the rank of a runtime tensor view when assigning!");
        }

        for (int i = buffer_info.ndim - 1; i >= 0; i--) {
            if (this->_dims[i] != buffer_info.shape[i]) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign buffer to runtime tensor view with different shapes!");
            }
        }

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            T *buffer_data = (T *)buffer_info.ptr;
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {
                size_t buffer_sent = 0, hold = sentinel, ord = 0;
                for (int i = 0; i < this->_rank; i++) {
                    ord += this->_strides[i] * (hold / this->_index_strides[i]);
                    buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_index_strides[i]);
                    hold %= this->_index_strides[i];
                }
                this->_data[ord] = buffer_data[buffer_sent];
            }
        } else {
            copy_and_cast_assign(buffer_info);
        }
        return *this;
    }

#define OPERATOR(OP, NAME, OPNAME)                                                                                                         \
    template <typename TOther>                                                                                                             \
    RuntimeTensorView<T> &operator OP(const TOther & other) {                                                                              \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OPNAME, other);                                                    \
    }                                                                                                                                      \
    template <typename TOther>                                                                                                             \
    RuntimeTensorView<T> &operator OP(const RuntimeTensor<TOther> &other) {                                                                \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OPNAME, other);                                                    \
    }                                                                                                                                      \
    template <typename TOther>                                                                                                             \
    RuntimeTensorView<T> &operator OP(const RuntimeTensorView<TOther> &other) {                                                            \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OPNAME, other);                                                    \
    }                                                                                                                                      \
    RuntimeTensorView<T> &operator OP(const pybind11::buffer & buffer) {                                                                   \
        pybind11::gil_scoped_acquire gil;                                                                                                  \
        pybind11::function           override = pybind11::get_override(static_cast<PyTensorView<T> *>(this), #OPNAME);                     \
                                                                                                                                           \
        if (override) {                                                                                                                    \
            auto o = override(buffer);                                                                                                     \
            if (pybind11::detail::cast_is_temporary_value_reference<RuntimeTensorView<T> &>::value) {                                      \
                static pybind11::detail::override_caster_t<RuntimeTensor<T> &> caster;                                                     \
                return pybind11::detail::cast_ref<RuntimeTensorView<T> &>(std::move(o), caster);                                           \
            }                                                                                                                              \
            return pybind11::detail::cast_safe<RuntimeTensorView<T> &>(std::move(o));                                                      \
        } else {                                                                                                                           \
            auto buffer_info = buffer.request();                                                                                           \
                                                                                                                                           \
            if (this->rank() != buffer_info.ndim) {                                                                                        \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform " #OP " with buffer object with different rank!");           \
            }                                                                                                                              \
            for (int i = buffer_info.ndim - 1; i >= 0; i--) {                                                                              \
                if (this->_dims[i] != buffer_info.shape[i]) {                                                                              \
                    EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform " #OP " with buffer object with different dimensions!");     \
                }                                                                                                                          \
            }                                                                                                                              \
                                                                                                                                           \
            if (buffer_info.item_type_is_equivalent_to<T>()) {                                                                             \
                T *buffer_data = (T *)buffer_info.ptr;                                                                                     \
                EINSUMS_OMP_PARALLEL_FOR                                                                                                   \
                for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {                                                           \
                    size_t buffer_sent = 0, hold = sentinel, ord = 0;                                                                      \
                    for (int i = 0; i < this->_rank; i++) {                                                                                \
                        ord += this->_strides[i] * (hold / this->_index_strides[i]);                                                       \
                        buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / this->_index_strides[i]);                 \
                        hold %= this->_index_strides[i];                                                                                   \
                    }                                                                                                                      \
                    this->_data[ord] OP buffer_data[buffer_sent];                                                                          \
                }                                                                                                                          \
            } else {                                                                                                                       \
                copy_and_cast_##NAME(buffer_info);                                                                                         \
            }                                                                                                                              \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
    }

    OPERATOR(*=, mult, operator*=)
    OPERATOR(/=, div, operator/=)
    OPERATOR(+=, add, operator+=)
    OPERATOR(-=, sub, operator-=)

#undef OPERATOR

    /**
     * @brief Get the dimension along a given axis.
     *
     * @param d The axis to query.
     */
    size_t dim(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, dim, d); }

    /**
     * @brief Get the dimensions of the view.
     */
    std::vector<size_t> dims() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensorView<T>, dims); }

    /**
     * @brief Get the stride along a given axis.
     *
     * @param d The axis to query.
     */
    size_t stride(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, stride, d); }

    /**
     * @brief Get the strides of the view.
     */
    std::vector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensorView<T>, strides); }

    /**
     * @brief Check whether the view sees all of the data of the tensor it views.
     */
    bool full_view_of_underlying() const noexcept override { PYBIND11_OVERRIDE(bool, RuntimeTensorView<T>, full_view_of_underlying); }

    /**
     * @brief Get the name of the tensor.
     */
    std::string const &name() const noexcept override { PYBIND11_OVERRIDE(std::string const &, RuntimeTensorView<T>, name); }

    /**
     * @brief Set the name of the tensor.
     *
     * @param new_name The new name for the tensor.
     */
    void set_name(std::string const &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_name, new_name); }

    /**
     * @brief Gets the rank of the tensor view.
     */
    size_t rank() const noexcept override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, rank); }
};

/**
 * @brief Expose runtime tensors to Python.
 *
 * @tparam T The stored type of the tensors to export.
 * @param mod The module which will contain the definitions.
 */
template <typename T>
void export_tensor(pybind11::module &mod) {
    std::string suffix = "";

    if constexpr (std::is_same_v<T, float>) {
        suffix = "F";
    } else if constexpr (std::is_same_v<T, double>) {
        suffix = "D";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        suffix = "C";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        suffix = "Z";
    }

    pybind11::class_<PyTensorIterator<T>, std::shared_ptr<PyTensorIterator<T>>>(mod, ("PyTensorIterator" + suffix).c_str())
        .def("__next__", &PyTensorIterator<T>::next, pybind11::return_value_policy::reference)
        .def("reversed", &PyTensorIterator<T>::reversed)
        .def("__iter__", [](PyTensorIterator<T> const &copy) { return copy; })
        .def("__reversed__", [](PyTensorIterator<T> const &copy) { return PyTensorIterator<T>(copy, true); });

    auto tensor_view =
        pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>, einsums::tensor_base::RuntimeTensorNoType>(
            mod, ("RuntimeTensorView" + suffix).c_str(), pybind11::buffer_protocol());
    pybind11::class_<RuntimeTensor<T>, PyTensor<T>, SharedRuntimeTensor<T>, einsums::tensor_base::RuntimeTensorNoType>(
        mod, ("RuntimeTensor" + suffix).c_str(), pybind11::buffer_protocol())
        .def(pybind11::init<>())
        .def(pybind11::init<std::string, std::vector<size_t> const &>())
        .def(pybind11::init<std::vector<size_t> const &>())
        .def(pybind11::init<pybind11::buffer const &>())
        .def("zero", &RuntimeTensor<T>::zero)
        .def("set_all", &RuntimeTensor<T>::set_all)
        .def("__getitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__getitem__",
             [](RuntimeTensor<T> &self, pybind11::slice const &args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__getitem__",
             [](RuntimeTensor<T> &self, int args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &key, T value) {
                 PyTensorView<T> cast(self);
                 cast.assign_values(value, key);
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &key, pybind11::buffer const &values) {
                 PyTensorView<T> cast(self);
                 cast.assign_values(values, key);
             })
#define OPERATOR(OP, TYPE)                                                                                                                 \
    .def(pybind11::self OP TYPE()).def(pybind11::self OP RuntimeTensor<TYPE>()).def(pybind11::self OP RuntimeTensorView<TYPE>())

            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
        .def(pybind11::self *= long())
        .def(
            "__imul__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self *= other; },
            pybind11::is_operator()) OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>)
            OPERATOR(/=, std::complex<double>)
        .def(pybind11::self /= long())
        .def(
            "__itruediv__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self /= other; },
            pybind11::is_operator()) OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>)
            OPERATOR(+=, std::complex<double>)
        .def(pybind11::self += long())
        .def(
            "__iadd__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self += other; },
            pybind11::is_operator()) OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>)
            OPERATOR(-=, std::complex<double>)
        .def(pybind11::self -= long())
        .def(
            "__isub__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self -= other; },
            pybind11::is_operator())
        .def("assign", [](PyTensor<T> &self, pybind11::buffer &buffer) { return self = buffer; })
        .def("dim", &RuntimeTensor<T>::dim)
        .def("dims", &RuntimeTensor<T>::dims)
        .def("stride", &RuntimeTensor<T>::stride)
        .def("strides", &RuntimeTensor<T>::strides)
        .def("to_rank_1_view", &RuntimeTensor<T>::to_rank_1_view)
        .def("get_name", &RuntimeTensor<T>::name)
        .def("set_name", &RuntimeTensor<T>::set_name)
        .def_property("name", &RuntimeTensor<T>::name, &RuntimeTensor<T>::set_name)
        .def("size", &RuntimeTensor<T>::size)
        .def("__len__", &RuntimeTensor<T>::size)
        .def("__iter__", [](RuntimeTensor<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](RuntimeTensor<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensor<T>::rank)
        .def("__copy__", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__deepcopy__", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("copy", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("deepcopy", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__str__",
             [](RuntimeTensor<T> const &self) {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def_buffer([](RuntimeTensor<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
    tensor_view.def(pybind11::init<>())
        .def(pybind11::init<RuntimeTensor<T> &>())
        .def(pybind11::init<RuntimeTensor<T> const &>())
        .def(pybind11::init<RuntimeTensorView<T> const &>())
        .def(pybind11::init<RuntimeTensor<T> &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensor<T> const &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensorView<T> &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensorView<T> const &, std::vector<size_t> const &>())
        .def(pybind11::init<pybind11::buffer &>())
        .def("zero", &RuntimeTensorView<T>::zero)
        .def("set_all", &RuntimeTensorView<T>::set_all)
        .def("__getitem__", [](PyTensorView<T> &self, pybind11::tuple const &args) { return self.subscript(args); })
        .def("__getitem__", [](PyTensorView<T> &self, pybind11::slice const &args) { return self.subscript(args); })
        .def("__getitem__", [](PyTensorView<T> &self, int args) { return self.subscript(args); })
        .def("__setitem__", [](PyTensorView<T> &self, pybind11::tuple const &key, T value) { self.assign_values(value, key); })
        .def("__setitem__",
             [](PyTensorView<T> &self, pybind11::tuple const &key, pybind11::buffer const &values) { self.assign_values(values, key); })
            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
        .def(pybind11::self *= long())
        .def(
            "__imul__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self *= other; }, pybind11::is_operator())
            OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>) OPERATOR(/=, std::complex<double>)
        .def(pybind11::self /= long())
        .def(
            "__itruediv__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self /= other; }, pybind11::is_operator())
            OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>) OPERATOR(+=, std::complex<double>)
        .def(pybind11::self += long())
        .def(
            "__iadd__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self += other; }, pybind11::is_operator())
            OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>) OPERATOR(-=, std::complex<double>)
        .def(pybind11::self -= long())
        .def(
            "__isub__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self -= other; }, pybind11::is_operator())
        .def("assign", [](PyTensorView<T> &self, pybind11::buffer &buffer) { return self = buffer; })
        .def("dim", &RuntimeTensorView<T>::dim)
        .def("dims", &RuntimeTensorView<T>::dims)
        .def("stride", &RuntimeTensorView<T>::stride)
        .def("strides", &RuntimeTensorView<T>::strides)
        .def("get_name", &RuntimeTensorView<T>::name)
        .def("set_name", &RuntimeTensorView<T>::set_name)
        .def_property("name", &RuntimeTensorView<T>::name, &RuntimeTensorView<T>::set_name)
        .def("size", &RuntimeTensorView<T>::size)
        .def("__len__", &RuntimeTensorView<T>::size)
        .def("__iter__", [](RuntimeTensorView<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](RuntimeTensorView<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensorView<T>::rank)
        .def("__str__",
             [](RuntimeTensorView<T> const &self) {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def_buffer([](RuntimeTensorView<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
#undef OPERATOR
}

/**
 * @brief Exposes extra symbols to Python that are not typed.
 *
 * @param mod The module to export to.
 */
EINSUMS_EXPORT void export_tensor_typeless(pybind11::module_ &mod);

} // namespace einsums::python