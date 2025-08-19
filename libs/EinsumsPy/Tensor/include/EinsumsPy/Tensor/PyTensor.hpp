//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorUtilities.hpp>

#include <memory>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <source_location>
#include <stdexcept>

#include "Einsums/TensorImpl/TensorImplOperations.hpp"

namespace einsums::python {

template <typename T>
einsums::detail::TensorImpl<T> buffer_to_tensor(pybind11::buffer &buffer) {
    pybind11::buffer_info info = buffer.request(true);

    if (!info.item_type_is_equivalent_to<T>()) {
        EINSUMS_THROW_EXCEPTION(pybind11::value_error, "The buffer format is not what is expected!");
    }

    BufferVector<size_t> strides(info.ndim);

    for (int i = 0; i < info.ndim; i++) {
        strides[i] = info.strides[i] / sizeof(T);
    }

    return einsums::detail::TensorImpl<T>(static_cast<T *>(info.ptr), info.shape, strides);
}

template <typename T>
einsums::detail::TensorImpl<T> const buffer_to_tensor(pybind11::buffer const &buffer) {
    pybind11::buffer_info info = buffer.request(false);

    if (!info.item_type_is_equivalent_to<T>()) {
        EINSUMS_THROW_EXCEPTION(pybind11::value_error, "The buffer format is not what is expected!");
    }

    BufferVector<size_t> strides(info.ndim);

    for (int i = 0; i < info.ndim; i++) {
        strides[i] = info.strides[i] / sizeof(T);
    }

    return einsums::detail::TensorImpl<T>(static_cast<T *>(info.ptr), info.shape, strides);
}

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
    BufferVector<size_t> _index_strides;

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

        BufferVector<size_t> ind(_tensor.rank());

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
        auto buffer_info = buffer.request(false);

        this->_data.resize(buffer_info.size);

        this->_impl = einsums::detail::TensorImpl<T>(this->_data.data(), buffer_info.shape);

        copy_and_cast_assign(buffer);
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
        BufferVector<ptrdiff_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<ptrdiff_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a value.
     *
     * @param args The index of the value.
     */
    T const &subscript_to_val(pybind11::tuple const &args) const {
        BufferVector<ptrdiff_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<ptrdiff_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a view
     *
     * @param args The index of the view. Can contain slices.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::tuple const &args) {
        BufferVector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                ptrdiff_t ind = pybind11::cast<ptrdiff_t>(arg);
                pass[i]       = RemovableRange{ind, ind};
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
        BufferVector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                ptrdiff_t ind = pybind11::cast<ptrdiff_t>(arg);
                pass[i]       = RemovableRange{ind, ind};
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
            if (args.size() < this->rank()) {
                return pybind11::cast(subscript_to_view(args));
            }
            if (args.size() > this->rank()) {
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
            if (index.size() < this->rank()) {
                assign_to_view(value, index);
                return pybind11::cast(subscript_to_view(index));
            }
            if (index.size() > this->rank()) {
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
            if (index.size() < this->rank()) {
                assign_to_view(value, index);
                return pybind11::cast(subscript_to_view(index));
            }
            if (index.size() > this->rank()) {
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

            BufferVector<Range> pass{Range{start, end}};

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

            BufferVector<Range> pass{Range{start, end}};

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

            BufferVector<Range> pass{Range{start, end}};

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
            if (this->rank() == 1) {
                return pybind11::cast(this->operator()(index));
            } else {
                return pybind11::cast(this->operator()(BufferVector<Range>{RemovableRange{index, index}}));
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
            if (this->rank() <= 1) {
                EINSUMS_THROW_EXCEPTION(std::length_error, "Can not assign buffer to a single position!");
            }

            return this->operator()(BufferVector<Range>{RemovableRange{index, index}}) = value;
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
            if (this->rank() <= 1) {
                T &target = this->operator()({index});
                target    = value;
                return pybind11::cast(target);
            }

            auto view = this->operator()(BufferVector<Range>{RemovableRange{index, index}});
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

        this->_data.resize(buffer_info.size);

        this->_impl = einsums::detail::TensorImpl<T>(this->_data.data(), buffer_info.shape);

        copy_and_cast_assign(buffer);

        return *this;
    }

  private:
#define COPY_CAST_OP(OP, NAME, FUNC)                                                                                                       \
    void copy_and_cast_##NAME(pybind11::buffer const &buffer) {                                                                            \
        pybind11::buffer_info buffer_info = buffer.request(false);                                                                         \
        auto                  format      = buffer_info.format;                                                                            \
        if (format.length() > 2) {                                                                                                         \
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can't handle user defined data type {}!", format);                              \
        }                                                                                                                                  \
        einsums::detail::TensorImpl<int8_t>                    int8tens;                                                                   \
        einsums::detail::TensorImpl<uint8_t>                   uint8tens;                                                                  \
        einsums::detail::TensorImpl<int16_t>                   int16tens;                                                                  \
        einsums::detail::TensorImpl<uint16_t>                  uint16tens;                                                                 \
        einsums::detail::TensorImpl<int32_t>                   int32tens;                                                                  \
        einsums::detail::TensorImpl<uint32_t>                  uint32tens;                                                                 \
        einsums::detail::TensorImpl<int64_t>                   int64tens;                                                                  \
        einsums::detail::TensorImpl<uint64_t>                  uint64tens;                                                                 \
        einsums::detail::TensorImpl<float>                     float_tens;                                                                 \
        einsums::detail::TensorImpl<double>                    double_tens;                                                                \
        einsums::detail::TensorImpl<long double>               ldouble_tens;                                                               \
        einsums::detail::TensorImpl<std::complex<float>>       cfloat_tens;                                                                \
        einsums::detail::TensorImpl<std::complex<double>>      cdouble_tens;                                                               \
        einsums::detail::TensorImpl<std::complex<long double>> cldouble_tens;                                                              \
        switch (format[0]) {                                                                                                               \
        case 'b':                                                                                                                          \
            int8tens = buffer_to_tensor<int8_t>(buffer);                                                                                   \
            FUNC(int8tens, this->impl());                                                                                                  \
            break;                                                                                                                         \
        case 'B':                                                                                                                          \
            uint8tens = buffer_to_tensor<uint8_t>(buffer);                                                                                 \
            FUNC(uint8tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'h':                                                                                                                          \
            int16tens = buffer_to_tensor<int16_t>(buffer);                                                                                 \
            FUNC(int16tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'H':                                                                                                                          \
            uint16tens = buffer_to_tensor<uint16_t>(buffer);                                                                               \
            FUNC(uint16tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'i':                                                                                                                          \
            int32tens = buffer_to_tensor<int32_t>(buffer);                                                                                 \
            FUNC(int32tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'I':                                                                                                                          \
            uint32tens = buffer_to_tensor<uint32_t>(buffer);                                                                               \
            FUNC(uint32tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'q':                                                                                                                          \
            int64tens = buffer_to_tensor<int64_t>(buffer);                                                                                 \
            FUNC(int64tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'Q':                                                                                                                          \
            uint64tens = buffer_to_tensor<uint64_t>(buffer);                                                                               \
            FUNC(uint64tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'l':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                int32tens = buffer_to_tensor<int32_t>(buffer);                                                                             \
                FUNC(int32tens, this->impl());                                                                                             \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                int64tens = buffer_to_tensor<int64_t>(buffer);                                                                             \
                FUNC(int64tens, this->impl());                                                                                             \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'L':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                uint32tens = buffer_to_tensor<uint32_t>(buffer);                                                                           \
                FUNC(uint32tens, this->impl());                                                                                            \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                uint64tens = buffer_to_tensor<uint64_t>(buffer);                                                                           \
                FUNC(uint64tens, this->impl());                                                                                            \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'f':                                                                                                                          \
            float_tens = buffer_to_tensor<float>(buffer);                                                                                  \
            assert(float_tens.data() == (float *)buffer_info.ptr);                                                                         \
            FUNC(float_tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'd':                                                                                                                          \
            double_tens = buffer_to_tensor<double>(buffer);                                                                                \
            assert(double_tens.data() == (double *)buffer_info.ptr);                                                                       \
            FUNC(double_tens, this->impl());                                                                                               \
            break;                                                                                                                         \
        case 'g':                                                                                                                          \
            ldouble_tens = buffer_to_tensor<long double>(buffer);                                                                          \
            FUNC(ldouble_tens, this->impl());                                                                                              \
            break;                                                                                                                         \
        case 'Z':                                                                                                                          \
            if constexpr (!IsComplexV<T>) {                                                                                                \
                EINSUMS_THROW_EXCEPTION(complex_conversion_error,                                                                          \
                                        "Can not cast complex to real! Perform your preferred cast before hand.");                         \
            } else {                                                                                                                       \
                switch (format[1]) {                                                                                                       \
                case 'f':                                                                                                                  \
                    cfloat_tens = buffer_to_tensor<std::complex<float>>(buffer);                                                           \
                    FUNC(cfloat_tens, this->impl());                                                                                       \
                    break;                                                                                                                 \
                case 'd':                                                                                                                  \
                    cdouble_tens = buffer_to_tensor<std::complex<double>>(buffer);                                                         \
                    FUNC(cdouble_tens, this->impl());                                                                                      \
                    break;                                                                                                                 \
                case 'g':                                                                                                                  \
                    cldouble_tens = buffer_to_tensor<std::complex<long double>>(buffer);                                                   \
                    FUNC(cldouble_tens, this->impl());                                                                                     \
                    break;                                                                                                                 \
                default:                                                                                                                   \
                    EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,             \
                                            pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                             \
                }                                                                                                                          \
            }                                                                                                                              \
            break;                                                                                                                         \
        default:                                                                                                                           \
            EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,                     \
                                    pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                                     \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign, einsums::detail::copy_to)
    COPY_CAST_OP(+=, add, einsums::detail::add_assign)
    COPY_CAST_OP(-=, sub, einsums::detail::sub_assign)
    COPY_CAST_OP(*=, mult, einsums::detail::mult_assign)
    COPY_CAST_OP(/=, div, einsums::detail::div_assign)
#undef COPY_CAST_OP

  public:
#define OPERATOR(OP, NAME, OPNAME)                                                                                                         \
    template <typename TOther>                                                                                                             \
    RuntimeTensor<T> &operator OP(TOther const &other) {                                                                                   \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OPNAME, other);                                                            \
    }                                                                                                                                      \
    template <typename TOther>                                                                                                             \
    RuntimeTensor<T> &operator OP(RuntimeTensor<TOther> const &other) {                                                                    \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OPNAME, other);                                                            \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    RuntimeTensor<T> &operator OP(RuntimeTensorView<TOther> const &other) {                                                                \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OPNAME, other);                                                            \
    }                                                                                                                                      \
    RuntimeTensor<T> &operator OP(pybind11::buffer const &buffer) {                                                                        \
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
            copy_and_cast_##NAME(buffer);                                                                                                  \
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
    BufferVector<size_t> dims() const noexcept override { PYBIND11_OVERRIDE(BufferVector<size_t>, RuntimeTensor<T>, dims); }

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
    BufferVector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(BufferVector<size_t>, RuntimeTensor<T>, strides); }

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

        if (!buffer_info.item_type_is_equivalent_to<T>()) {
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not create RuntimeTensorView from buffer whose type does not match!");
        }

        BufferVector<size_t> strides(buffer_info.ndim);

        for (int i = 0; i < strides.size(); i++) {
            strides[i] = buffer_info.strides[i] / sizeof(T);
        }

        this->_impl = einsums::detail::TensorImpl<T>(static_cast<T *>(buffer_info.ptr), buffer_info.shape, strides);
    }

    /**
     * @brief Create a view of the given buffer.
     *
     * @param buffer The buffer to view.
     */
    PyTensorView(pybind11::buffer const &buffer) {
        pybind11::buffer_info buffer_info = buffer.request(false);

        if (!buffer_info.item_type_is_equivalent_to<T>()) {
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not create RuntimeTensorView from buffer whose type does not match!");
        }

        BufferVector<size_t> strides(buffer_info.ndim);

        for (int i = 0; i < strides.size(); i++) {
            strides[i] = buffer_info.strides[i] / sizeof(T);
        }

        this->_impl = einsums::detail::TensorImpl<T>(static_cast<T *>(buffer_info.ptr), buffer_info.shape, strides);
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
        BufferVector<ptrdiff_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<ptrdiff_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Worker method that subscripts into the view and returns a reference to the requested element.
     *
     * @param args The indices to use for the subscript.
     */
    T const &subscript_to_val(pybind11::tuple const &args) const {
        BufferVector<ptrdiff_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            pass[i] = pybind11::cast<ptrdiff_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Worker method that creates a view based on the indices and slices passed in.
     *
     * @param args The indices and slices to use for view creation.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::tuple const &args) {
        BufferVector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                ptrdiff_t ind = pybind11::cast<ptrdiff_t>(arg);
                pass[i]       = RemovableRange{ind, ind};
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
        BufferVector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                ptrdiff_t ind = pybind11::cast<ptrdiff_t>(arg);
                pass[i]       = RemovableRange{ind, ind};
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
    RuntimeTensorView<T> subscript_to_view(pybind11::slice const &args) {
        Range pass;

        size_t start, stop, step, slice_length;
        (pybind11::cast<pybind11::slice>(args)).compute(this->dim(0), &start, &stop, &step, &slice_length);
        if (step != 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with step sizes other than 1!");
        }
        pass = Range{start, stop};

        return this->operator()(pass);
    }

    /**
     * @brief Worker method that creates a view based on the indices and slices passed in.
     *
     * @param args The indices and slices to use for view creation.
     */
    RuntimeTensorView<T> subscript_to_view(pybind11::slice const &args) const {
        Range pass;

        size_t start, stop, step, slice_length;
        (pybind11::cast<pybind11::slice>(args)).compute(this->dim(0), &start, &stop, &step, &slice_length);
        if (step != 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with step sizes other than 1!");
        }
        pass = Range{start, stop};

        return this->operator()(pass);
    }

    /**
     * @brief Set the value at the given point in the tensor to the given value.
     *
     * @param value The new value.
     * @param index Where to set the value.
     */
    void set_value_at(T value, std::vector<ptrdiff_t> const &index) {
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

    /**
     * @brief Copy the data from a buffer into this view.
     *
     * Creates a view of part of the tensor using the subscript arguments, then
     * assigns the buffer to that view.
     *
     * @param view The buffer to copy.
     * @param args The position to copy to.
     */
    void assign_to_view(pybind11::buffer const &view, pybind11::slice const &args) {
        PyTensorView<T> this_view = subscript_to_view(args);

        this_view = view;
    }

    /**
     * @brief Fill part of the view with the given value.
     *
     * @param value The value to fill the view with.
     * @param args Indices and slices that determine the part of the view to fill.
     */
    void assign_to_view(T value, pybind11::slice const &args) {
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
            if (args.size() < this->rank()) {
                return pybind11::cast(subscript_to_view(args));
            }
            if (args.size() > this->rank()) {
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
            if (index.size() < this->rank()) {
                assign_to_view(value, index);
                return pybind11::cast(*this);
            }
            if (index.size() > this->rank()) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                auto const &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    pybind11::object out = this->subscript(index);
                    return out;
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
            if (index.size() < this->rank()) {
                assign_to_view(value, index);
                return pybind11::cast(*this);
            }
            if (index.size() > this->rank()) {
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

            pybind11::cast<pybind11::slice>(arg).compute(this->dim(0), &start, &end, &step, &length);

            if (step != 1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Can not handle slices with steps not equal to 1!");
            }

            BufferVector<Range> pass{Range{start, end}};

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
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<RuntimeTensorView<T>>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<RuntimeTensorView<T>>(std::move(o));
        } else {
            assign_to_view(value, index);
            RuntimeTensorView<T> out = this->subscript(index);
            return out;
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

            BufferVector<Range> pass{Range{start, end}};

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
            if (this->rank() == 1) {
                return pybind11::cast(this->operator()(index));
            } else {
                return pybind11::cast(this->operator()(BufferVector<Range>{RemovableRange{index, index}}));
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
            if (this->rank() <= 1) {
                EINSUMS_THROW_EXCEPTION(std::length_error, "Can not assign buffer to a single position!");
            }

            return this->operator()(BufferVector<Range>{RemovableRange{index, index}}) = value;
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
            if (this->rank() <= 1) {
                T &target = this->operator()({index});
                target    = value;
                return pybind11::cast(target);
            }

            auto view = this->operator()(BufferVector<Range>{RemovableRange{index, index}});
            view      = value;
            return pybind11::cast(view);
        }
    }

  private:
#define COPY_CAST_OP(OP, NAME, FUNC)                                                                                                       \
    void copy_and_cast_##NAME(pybind11::buffer const &buffer) {                                                                            \
        pybind11::buffer_info buffer_info = buffer.request(false);                                                                         \
        auto                  format      = buffer_info.format;                                                                            \
        if (format.length() > 2) {                                                                                                         \
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can't handle user defined data type {}!", format);                              \
        }                                                                                                                                  \
        einsums::detail::TensorImpl<int8_t>                    int8tens;                                                                   \
        einsums::detail::TensorImpl<uint8_t>                   uint8tens;                                                                  \
        einsums::detail::TensorImpl<int16_t>                   int16tens;                                                                  \
        einsums::detail::TensorImpl<uint16_t>                  uint16tens;                                                                 \
        einsums::detail::TensorImpl<int32_t>                   int32tens;                                                                  \
        einsums::detail::TensorImpl<uint32_t>                  uint32tens;                                                                 \
        einsums::detail::TensorImpl<int64_t>                   int64tens;                                                                  \
        einsums::detail::TensorImpl<uint64_t>                  uint64tens;                                                                 \
        einsums::detail::TensorImpl<float>                     float_tens;                                                                 \
        einsums::detail::TensorImpl<double>                    double_tens;                                                                \
        einsums::detail::TensorImpl<long double>               ldouble_tens;                                                               \
        einsums::detail::TensorImpl<std::complex<float>>       cfloat_tens;                                                                \
        einsums::detail::TensorImpl<std::complex<double>>      cdouble_tens;                                                               \
        einsums::detail::TensorImpl<std::complex<long double>> cldouble_tens;                                                              \
        switch (format[0]) {                                                                                                               \
        case 'b':                                                                                                                          \
            int8tens = buffer_to_tensor<int8_t>(buffer);                                                                                   \
            FUNC(int8tens, this->impl());                                                                                                  \
            break;                                                                                                                         \
        case 'B':                                                                                                                          \
            uint8tens = buffer_to_tensor<uint8_t>(buffer);                                                                                 \
            FUNC(uint8tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'h':                                                                                                                          \
            int16tens = buffer_to_tensor<int16_t>(buffer);                                                                                 \
            FUNC(int16tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'H':                                                                                                                          \
            uint16tens = buffer_to_tensor<uint16_t>(buffer);                                                                               \
            FUNC(uint16tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'i':                                                                                                                          \
            int32tens = buffer_to_tensor<int32_t>(buffer);                                                                                 \
            FUNC(int32tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'I':                                                                                                                          \
            uint32tens = buffer_to_tensor<uint32_t>(buffer);                                                                               \
            FUNC(uint32tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'q':                                                                                                                          \
            int64tens = buffer_to_tensor<int64_t>(buffer);                                                                                 \
            FUNC(int64tens, this->impl());                                                                                                 \
            break;                                                                                                                         \
        case 'Q':                                                                                                                          \
            uint64tens = buffer_to_tensor<uint64_t>(buffer);                                                                               \
            FUNC(uint64tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'l':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                int32tens = buffer_to_tensor<int32_t>(buffer);                                                                             \
                FUNC(int32tens, this->impl());                                                                                             \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                int64tens = buffer_to_tensor<int64_t>(buffer);                                                                             \
                FUNC(int64tens, this->impl());                                                                                             \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'L':                                                                                                                          \
            if (buffer_info.itemsize == 4) {                                                                                               \
                uint32tens = buffer_to_tensor<uint32_t>(buffer);                                                                           \
                FUNC(uint32tens, this->impl());                                                                                            \
            } else if (buffer_info.itemsize == 8) {                                                                                        \
                uint64tens = buffer_to_tensor<uint64_t>(buffer);                                                                           \
                FUNC(uint64tens, this->impl());                                                                                            \
            } else {                                                                                                                       \
                EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                                \
                                        "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");                    \
            }                                                                                                                              \
            break;                                                                                                                         \
        case 'f':                                                                                                                          \
            float_tens = buffer_to_tensor<float>(buffer);                                                                                  \
            FUNC(float_tens, this->impl());                                                                                                \
            break;                                                                                                                         \
        case 'd':                                                                                                                          \
            double_tens = buffer_to_tensor<double>(buffer);                                                                                \
            FUNC(double_tens, this->impl());                                                                                               \
            break;                                                                                                                         \
        case 'g':                                                                                                                          \
            ldouble_tens = buffer_to_tensor<long double>(buffer);                                                                          \
            FUNC(ldouble_tens, this->impl());                                                                                              \
            break;                                                                                                                         \
        case 'Z':                                                                                                                          \
            if constexpr (!IsComplexV<T>) {                                                                                                \
                EINSUMS_THROW_EXCEPTION(complex_conversion_error,                                                                          \
                                        "Can not cast complex to real! Perform your preferred cast before hand.");                         \
            } else {                                                                                                                       \
                switch (format[1]) {                                                                                                       \
                case 'f':                                                                                                                  \
                    cfloat_tens = buffer_to_tensor<std::complex<float>>(buffer);                                                           \
                    FUNC(cfloat_tens, this->impl());                                                                                       \
                    break;                                                                                                                 \
                case 'd':                                                                                                                  \
                    cdouble_tens = buffer_to_tensor<std::complex<double>>(buffer);                                                         \
                    FUNC(cdouble_tens, this->impl());                                                                                      \
                    break;                                                                                                                 \
                case 'g':                                                                                                                  \
                    cldouble_tens = buffer_to_tensor<std::complex<long double>>(buffer);                                                   \
                    FUNC(cldouble_tens, this->impl());                                                                                     \
                    break;                                                                                                                 \
                default:                                                                                                                   \
                    EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,             \
                                            pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                             \
                }                                                                                                                          \
            }                                                                                                                              \
            break;                                                                                                                         \
        default:                                                                                                                           \
            EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Can not convert format descriptor {} to {} ({})!", format,                     \
                                    pybind11::type_id<T>(), pybind11::format_descriptor<T>::format());                                     \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign, einsums::detail::copy_to)
    COPY_CAST_OP(+=, add, einsums::detail::add_assign)
    COPY_CAST_OP(-=, sub, einsums::detail::sub_assign)
    COPY_CAST_OP(*=, mult, einsums::detail::mult_assign)
    COPY_CAST_OP(/=, div, einsums::detail::div_assign)
#undef COPY_CAST_OP

  public:
#define OPERATOR(OP, NAME, OPNAME)                                                                                                         \
    template <typename TOther>                                                                                                             \
    RuntimeTensorView<T> &operator OP(TOther const &other) {                                                                               \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OPNAME, other);                                                    \
    }                                                                                                                                      \
    template <typename TOther>                                                                                                             \
    RuntimeTensorView<T> &operator OP(RuntimeTensor<TOther> const &other) {                                                                \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OPNAME, other);                                                    \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    RuntimeTensorView<T> &operator OP(RuntimeTensorView<TOther> const &other) {                                                            \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OPNAME, other);                                                    \
    }                                                                                                                                      \
    RuntimeTensorView<T> &operator OP(pybind11::buffer const &buffer) {                                                                    \
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
            if (buffer_info.ndim == 0) {                                                                                                   \
                if (buffer_info.format.length() == 0 || buffer_info.format.length() > 2) {                                                 \
                    EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Could not handle user defined buffer format \"{}\"!",                  \
                                            buffer_info.format);                                                                           \
                }                                                                                                                          \
                switch (buffer_info.format[0]) {                                                                                           \
                case 'b':                                                                                                                  \
                    *this OP einsums::detail::convert<int8_t, T>(*static_cast<int8_t *>(buffer_info.ptr));                                 \
                    break;                                                                                                                 \
                case 'B':                                                                                                                  \
                    *this OP einsums::detail::convert<uint8_t, T>(*static_cast<uint8_t *>(buffer_info.ptr));                               \
                    break;                                                                                                                 \
                case 'h':                                                                                                                  \
                    *this OP einsums::detail::convert<int16_t, T>(*static_cast<int16_t *>(buffer_info.ptr));                               \
                    break;                                                                                                                 \
                case 'H':                                                                                                                  \
                    *this OP einsums::detail::convert<uint16_t, T>(*static_cast<uint16_t *>(buffer_info.ptr));                             \
                    break;                                                                                                                 \
                case 'i':                                                                                                                  \
                    *this OP einsums::detail::convert<int32_t, T>(*static_cast<int32_t *>(buffer_info.ptr));                               \
                    break;                                                                                                                 \
                case 'I':                                                                                                                  \
                    *this OP einsums::detail::convert<uint32_t, T>(*static_cast<uint32_t *>(buffer_info.ptr));                             \
                    break;                                                                                                                 \
                case 'q':                                                                                                                  \
                    *this OP einsums::detail::convert<int64_t, T>(*static_cast<int64_t *>(buffer_info.ptr));                               \
                    break;                                                                                                                 \
                case 'Q':                                                                                                                  \
                    *this OP einsums::detail::convert<uint64_t, T>(*static_cast<uint64_t *>(buffer_info.ptr));                             \
                    break;                                                                                                                 \
                case 'l':                                                                                                                  \
                    if (buffer_info.itemsize == 4) {                                                                                       \
                        *this OP einsums::detail::convert<int32_t, T>(*static_cast<int32_t *>(buffer_info.ptr));                           \
                    } else if (buffer_info.itemsize == 8) {                                                                                \
                        *this OP einsums::detail::convert<int64_t, T>(*static_cast<int64_t *>(buffer_info.ptr));                           \
                    } else {                                                                                                               \
                        EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                        \
                                                "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");            \
                    }                                                                                                                      \
                    break;                                                                                                                 \
                case 'L':                                                                                                                  \
                    if (buffer_info.itemsize == 4) {                                                                                       \
                        *this OP einsums::detail::convert<uint32_t, T>(*static_cast<uint32_t *>(buffer_info.ptr));                         \
                    } else if (buffer_info.itemsize == 8) {                                                                                \
                        *this OP einsums::detail::convert<uint64_t, T>(*static_cast<uint64_t *>(buffer_info.ptr));                         \
                    } else {                                                                                                               \
                        EINSUMS_THROW_EXCEPTION(std::runtime_error,                                                                        \
                                                "Something's wrong with your system! Python ints are neither 32 nor 64 bits!");            \
                    }                                                                                                                      \
                    break;                                                                                                                 \
                case 'f':                                                                                                                  \
                    *this OP einsums::detail::convert<float, T>(*static_cast<float *>(buffer_info.ptr));                                   \
                    break;                                                                                                                 \
                case 'd':                                                                                                                  \
                    *this OP einsums::detail::convert<double, T>(*static_cast<double *>(buffer_info.ptr));                                 \
                    break;                                                                                                                 \
                case 'g':                                                                                                                  \
                    *this OP einsums::detail::convert<long double, T>(*static_cast<long double *>(buffer_info.ptr));                       \
                    break;                                                                                                                 \
                case 'Z':                                                                                                                  \
                    if constexpr (!IsComplexV<T>) {                                                                                        \
                        EINSUMS_THROW_EXCEPTION(complex_conversion_error,                                                                  \
                                                "Can not cast complex to real! Perform your preferred cast before hand.");                 \
                    } else {                                                                                                               \
                        switch (buffer_info.format[1]) {                                                                                   \
                        case 'f':                                                                                                          \
                            *this OP einsums::detail::convert<std::complex<float>, T>(                                                     \
                                *static_cast<std::complex<float> *>(buffer_info.ptr));                                                     \
                            break;                                                                                                         \
                        case 'd':                                                                                                          \
                            *this OP einsums::detail::convert<std::complex<double>, T>(                                                    \
                                *static_cast<std::complex<double> *>(buffer_info.ptr));                                                    \
                            break;                                                                                                         \
                        case 'g':                                                                                                          \
                            *this OP einsums::detail::convert<std::complex<long double>, T>(                                               \
                                *static_cast<std::complex<long double> *>(buffer_info.ptr));                                               \
                            break;                                                                                                         \
                        default:                                                                                                           \
                            EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Could not handle complex buffer format \"{}\"!",               \
                                                    buffer_info.format);                                                                   \
                        }                                                                                                                  \
                    }                                                                                                                      \
                default:                                                                                                                   \
                    EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Could not handle user-defined buffer format \"{}\"!",                  \
                                            buffer_info.format);                                                                           \
                }                                                                                                                          \
                return *this;                                                                                                              \
            }                                                                                                                              \
                                                                                                                                           \
            if (this->rank() != buffer_info.ndim) {                                                                                        \
                EINSUMS_THROW_EXCEPTION(tensor_compat_error,                                                                               \
                                        "Can not perform " #OP " with buffer object with different rank! Got {}, needed {}.",              \
                                        buffer_info.ndim, this->rank());                                                                   \
            }                                                                                                                              \
            copy_and_cast_##NAME(buffer);                                                                                                  \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
    }

    OPERATOR(=, assign, operator=)
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
    BufferVector<size_t> dims() const noexcept override { PYBIND11_OVERRIDE(BufferVector<size_t>, RuntimeTensorView<T>, dims); }

    /**
     * @brief Get the stride along a given axis.
     *
     * @param d The axis to query.
     */
    size_t stride(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, stride, d); }

    /**
     * @brief Get the strides of the view.
     */
    BufferVector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(BufferVector<size_t>, RuntimeTensorView<T>, strides); }

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
 * @brief Exposes extra symbols to Python that are not typed.
 *
 * @param mod The module to export to.
 */
EINSUMS_EXPORT void export_tensor_typeless(pybind11::module_ &mod);

} // namespace einsums::python