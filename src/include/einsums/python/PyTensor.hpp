#pragma once

#include "einsums/RuntimeTensor.hpp"
#include "einsums/utility/IndexUtils.hpp"

#include <memory>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace einsums::python {

// Forward declarations
template <typename T>
class PyTensor;

template <typename T>
using SharedRuntimeTensor = std::shared_ptr<RuntimeTensor<T>>;

template <typename T>
class PyTensorView;

template <typename T>
using SharedRuntimeTensorView = std::shared_ptr<RuntimeTensorView<T>>;

/**
 * @struct PyTensorIterator<T>
 *
 * @brief Walks through the elements of a tensor.
 *
 * @tparam T The type stored in the tensor.
 */
template <typename T>
class EINSUMS_EXPORT PyTensorIterator {
  private:
    std::mutex           _lock;
    size_t               _curr_index, _elements;
    std::vector<size_t>  _index_strides;
    RuntimeTensorView<T> _tensor;
    bool                 _stop{false}, _reverse{false};

  public:
    PyTensorIterator(const PyTensorIterator &copy, bool reverse = false)
        : _curr_index{copy._curr_index}, _elements{copy._elements},
          _index_strides(copy._index_strides.cbegin(), copy._index_strides.cend()), _tensor(copy._tensor),
          _stop{copy._curr_index < 0 || copy._curr_index >= copy._elements}, _reverse{reverse != copy._reverse} {}

    /**
     * Create an iterator around a tensor. Can be reversed.
     *
     * @param other The tensor to walk through.
     * @param reverse Whether to go forward or backward.
     */
    PyTensorIterator(const RuntimeTensor<T> &other, bool reverse = false)
        : _tensor{other}, _reverse{reverse}, _index_strides(other.rank()), _elements(other.size()) {
        if (!reverse) {
            _curr_index = 0;
        } else {
            _curr_index = other.size() - 1;
        }

        tensor_algebra::detail::dims_to_strides(other.dims(), _index_strides);
    }

    /**
     * Create an iterator around a tensor view. Can be reversed.
     *
     * @param other The tensor view to walk through.
     * @param reverse Whether to go forward or backward.
     */
    PyTensorIterator(const RuntimeTensorView<T> &other, bool reverse = false)
        : _tensor{other}, _reverse{reverse}, _index_strides(other.rank()), _elements(other.size()) {
        if (!reverse) {
            _curr_index = 0;
        } else {
            _curr_index = other.size() - 1;
        }

        tensor_algebra::detail::dims_to_strides(other.dims(), _index_strides);
    }

    /**
     * Get the next element in the tensor.
     */
    T next() THROWS(pybind11::stop_iteration) {
        auto guard = std::lock_guard(_lock);

        if (_stop) {
            throw pybind11::stop_iteration();
        }

        std::vector<size_t> ind;

        tensor_algebra::detail::sentinel_to_indices(_curr_index, _index_strides, ind);

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

template <typename T>
class PyTensor : public RuntimeTensor<T> {
  public:
    using RuntimeTensor<T>::RuntimeTensor;

    /**
     * @brief Copy constructor from shared pointer.
     */
    PyTensor(const std::shared_ptr<PyTensor<T>> &other) : PyTensor(*other) {}

    /**
     * Create a tensor from a Python buffer object.
     */
    PyTensor(const pybind11::buffer &buffer) THROWS(einsums::EinsumsException) {
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
    T &subscript_to_val(const pybind11::tuple &args) {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a value.
     *
     * @param args The index of the value.
     */
    const T &subscript_to_val(const pybind11::tuple &args) const {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    /**
     * @brief Subscript the tensor to get a view
     *
     * @param args The index of the view. Can contain slices.
     */
    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) THROWS(einsums::EinsumsException) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
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
    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) const THROWS(einsums::EinsumsException) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    /**
     * @brief Set the value of a certain position.
     *
     * @param index The index of the position.
     */
    void set_value_at(T value, const std::vector<ptrdiff_t> &index) {
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
    void assign_to_view(const pybind11::buffer &view, const pybind11::tuple &args) THROWS(einsums::EinsumsException) {
        PyTensorView<T> this_view = subscript_to_view(args);

        this_view = view;
    }

    /**
     * @brief Copy one value to multiple positions.
     *
     * @param value The value to set.
     * @param args The indices to set. Can contain slices.
     */
    void assign_to_view(T value, const pybind11::tuple &args) THROWS(einsums::EinsumsException) {
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
    pybind11::object subscript(const pybind11::tuple &args) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__subscript");

        if (override) {
            auto o = override(args);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11 ::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11 ::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11 ::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11 ::object>(std ::move(o));
        } else {
            if (args.size() < this->_rank) {
                return pybind11::cast(subscript_to_view(args));
            }
            if (args.size() > this->_rank) {
                throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
            }
            for (int i = 0; i < args.size(); i++) {
                const auto &arg = args[i];

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
     * @param args The indices to use to subscript.
     * @return A Python object containing a reference to a single value or a RuntimeTensorView.
     */
    pybind11::object assign_values(const pybind11::buffer &value, const pybind11::tuple &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(subscript_to_view(index));
            }
            if (index.size() > this->_rank) {
                throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                const auto &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    return subscript(index);
                }
            }
            throw EINSUMSEXCEPTION("Can not assign buffer object to a single position!");
        }
    }

    /**
     * @brief Copy a single value over a section of the tensor.
     *
     * @param value The value to copy.
     * @param args The indices to use to subscript.
     * @return A Python object containing a reference to a single value or a RuntimeTensorView.
     */
    pybind11::object assign_values(T value, const pybind11::tuple &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(subscript_to_view(index));
            }
            if (index.size() > this->_rank) {
                throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                const auto &arg = index[i];

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
    RuntimeTensorView<T> subscript(const pybind11::slice &arg) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__subscript");

        if (override) {
            auto o = override(arg);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(arg).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
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
    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, const pybind11::slice &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
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
    RuntimeTensorView<T> assign_values(T value, const pybind11::slice &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
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
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
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
    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, ptrdiff_t index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            if (this->_rank <= 1) {
                throw EINSUMSEXCEPTION("Can not assign buffer to a single position!");
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
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
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
    RuntimeTensor<T> &operator=(const pybind11::buffer &buffer) THROWS(einsums::EinsumsException) {
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
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
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
            throw EINSUMSEXCEPTION("Can't handle most user defined data type " + format + "!");                                            \
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
                throw EINSUMSEXCEPTION("Can not convert format descriptor " + format + " to " + pybind11::type_id<T>() + " (" +            \
                                       pybind11::format_descriptor<T>::format() + ")!");                                                   \
            }                                                                                                                              \
            break;                                                                                                                         \
        default:                                                                                                                           \
            throw EINSUMSEXCEPTION("Can not convert format descriptor " + format + " to " + pybind11::type_id<T>() + " (" +                \
                                   pybind11::format_descriptor<T>::format() + ")!");                                                       \
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
    RuntimeTensor<T> &operator OP(const pybind11::buffer & buffer) THROWS(einsums::EinsumsException) {                                     \
        pybind11::gil_scoped_acquire gil;                                                                                                  \
        pybind11::function           override = pybind11::get_override(static_cast<PyTensor<T> *>(this), #OPNAME);                         \
                                                                                                                                           \
        if (override) {                                                                                                                    \
            auto o = override(buffer);                                                                                                     \
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensor<T> &>::value) {                                        \
                static pybind11 ::detail ::override_caster_t<RuntimeTensor<T> &> caster;                                                   \
                return pybind11 ::detail ::cast_ref<RuntimeTensor<T> &>(std ::move(o), caster);                                            \
            }                                                                                                                              \
            return pybind11 ::detail ::cast_safe<RuntimeTensor<T> &>(std ::move(o));                                                       \
        } else {                                                                                                                           \
            auto buffer_info = buffer.request();                                                                                           \
                                                                                                                                           \
            if (this->rank() != buffer_info.ndim) {                                                                                        \
                throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different rank!");                                 \
            }                                                                                                                              \
                                                                                                                                           \
            bool is_view = false;                                                                                                          \
            for (int i = buffer_info.ndim - 1; i >= 0; i--) {                                                                              \
                if (this->_dims[i] != buffer_info.shape[i]) {                                                                              \
                    throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different dimensions!");                       \
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
                    EINSUMS_OMP_PARALLEL_FOR                                                                                               \
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

    size_t dim(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, dim, d); }

    std::vector<size_t> dims() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensor<T>, dims); }

    const typename RuntimeTensor<T>::Vector &vector_data() const noexcept override {
        PYBIND11_OVERRIDE(const typename RuntimeTensor<T>::Vector &, RuntimeTensor<T>, vector_data);
    }

    typename RuntimeTensor<T>::Vector &vector_data() noexcept override {
        PYBIND11_OVERRIDE(typename RuntimeTensor<T>::Vector &, RuntimeTensor<T>, vector_data);
    }

    size_t stride(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, stride, d); }

    std::vector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensor<T>, strides); }

    RuntimeTensorView<T> to_rank_1_view() const override { PYBIND11_OVERRIDE(RuntimeTensorView<T>, RuntimeTensor<T>, to_rank_1_view); }

    bool full_view_of_underlying() const noexcept override { PYBIND11_OVERRIDE(bool, RuntimeTensor<T>, full_view_of_underlying); }

    const std::string &name() const noexcept override { PYBIND11_OVERRIDE_NAME(const std::string &, RuntimeTensor<T>, "get_name", name); }

    void set_name(const std::string &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, set_name, new_name); }

    size_t rank() const noexcept override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, rank); }
};

/**
 * @class PyTensorView<T>
 *
 * @brief Views a runtime tensor and makes it available to Python.
 *
 * @see PyTensor<T> for information on methods.
 */
template <typename T>
class PyTensorView : public RuntimeTensorView<T> {
  public:
    using RuntimeTensorView<T>::RuntimeTensorView;

    PyTensorView(const PyTensorView<T> &) = default;
    PyTensorView(const RuntimeTensorView<T> &copy) : RuntimeTensorView<T>(copy) {}

    PyTensorView(pybind11::buffer &buffer) THROWS(einsums::EinsumsException) {
        pybind11::buffer_info buffer_info = buffer.request(true);

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            this->_data = (T *)buffer_info.ptr;
        } else {
            throw EINSUMSEXCEPTION("Can not create RuntimeTensorView from buffer whose type does not match!");
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

    void zero() override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, zero); }

    void set_all(T val) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_all, val); }

  private:
    T &subscript_to_val(const pybind11::tuple &args) {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    const T &subscript_to_val(const pybind11::tuple &args) const {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) THROWS(einsums::EinsumsException) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) const THROWS(einsums::EinsumsException) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    void set_value_at(T value, const std::vector<size_t> &index) {
        T &target = this->operator()(index);
        target    = value;
        return target;
    }

    void assign_to_view(const pybind11::buffer &view, const pybind11::tuple &args) {
        PyTensorView<T> this_view = subscript_to_view(args);

        this_view = view;
    }

    void assign_to_view(T value, const pybind11::tuple &args) {
        auto this_view = subscript_to_view(args);

        this_view = value;
    }

  public:
    pybind11::object subscript(const pybind11::tuple &args) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__subscript");

        if (override) {
            auto o = override(args);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11 ::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11 ::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11 ::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11 ::object>(std ::move(o));
        } else {
            if (args.size() < this->_rank) {
                return pybind11::cast(subscript_to_view(args));
            }
            if (args.size() > this->_rank) {
                throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
            }
            for (int i = 0; i < args.size(); i++) {
                const auto &arg = args[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    return pybind11::cast(subscript_to_view(args));
                }
            }

            return pybind11::cast(subscript_to_val(args));
        }
    }

    pybind11::object assign_values(const pybind11::buffer &value, const pybind11::tuple &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(*this);
            }
            if (index.size() > this->_rank) {
                throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                const auto &arg = index[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    assign_to_view(value, index);
                    return subscript(index);
                }
            }
            throw EINSUMSEXCEPTION("Can not assign buffer object to a single position!");
        }
    }

    pybind11::object assign_values(T value, const pybind11::tuple &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
        } else {
            if (index.size() < this->_rank) {
                assign_to_view(value, index);
                return pybind11::cast(*this);
            }
            if (index.size() > this->_rank) {
                throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
            }
            for (int i = 0; i < index.size(); i++) {
                const auto &arg = index[i];

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

    RuntimeTensorView<T> subscript(const pybind11::slice &arg) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__subscript");

        if (override) {
            auto o = override(arg);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(arg).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass);
        }
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, const pybind11::slice &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass) = value;
        }
    }

    RuntimeTensorView<T> assign_values(T value, const pybind11::slice &index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            size_t start, end, step, length;

            pybind11::cast<pybind11::slice>(index).compute(this->_dims[0], &start, &end, &step, &length);

            if (step != 1) {
                throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
            }

            std::vector<Range> pass{Range{start, end}};

            return this->operator()(pass) = value;
        }
    }

    pybind11::object subscript(ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__subscript");

        if (override) {
            auto o = override(index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
        } else {
            if (this->_rank == 1) {
                return pybind11::cast(this->operator()(index));
            } else {
                return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
            }
        }
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, ptrdiff_t index) THROWS(einsums::EinsumsException) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T>>::value) {
                static pybind11 ::detail ::override_caster_t<RuntimeTensorView<T>> caster;
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T>>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T>>(std ::move(o));
        } else {
            if (this->_rank <= 1) {
                throw EINSUMSEXCEPTION("Can not assign buffer to a single position!");
            }

            return this->operator()(std::vector<Range>{Range{-1, index}}) = value;
        }
    }

    pybind11::object assign_values(T value, ptrdiff_t index) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeTensorView<T> *>(this), "__assign");

        if (override) {
            auto o = override(value, index);
            if (pybind11 ::detail ::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11 ::detail ::override_caster_t<pybind11::object> caster;
                return pybind11 ::detail ::cast_ref<pybind11::object>(std ::move(o), caster);
            }
            return pybind11 ::detail ::cast_safe<pybind11::object>(std ::move(o));
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
            throw EINSUMSEXCEPTION("Can't handle most user defined data type " + format + "!");                                            \
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
                throw EINSUMSEXCEPTION("Can not convert format descriptor " + format + " to " + pybind11::type_id<T>() + " (" +            \
                                       pybind11::format_descriptor<T>::format() + ")!");                                                   \
            }                                                                                                                              \
            break;                                                                                                                         \
        default:                                                                                                                           \
            throw EINSUMSEXCEPTION("Can not convert format descriptor " + format + " to " + pybind11::type_id<T>() + " (" +                \
                                   pybind11::format_descriptor<T>::format() + ")!");                                                       \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign)
    COPY_CAST_OP(+=, add)
    COPY_CAST_OP(-=, sub)
    COPY_CAST_OP(*=, mult)
    COPY_CAST_OP(/=, div)
#undef COPY_CAST_OP

  public:
    PyTensorView<T> &operator=(const pybind11::buffer &buffer) THROWS(einsums::EinsumsException) {
        auto buffer_info = buffer.request();

        if (this->rank() != buffer_info.ndim) {
            throw EINSUMSEXCEPTION("Can not change the rank of a runtime tensor view when assigning!");
        }

        for (int i = buffer_info.ndim - 1; i >= 0; i--) {
            if (this->_dims[i] != buffer_info.shape[i]) {
                throw EINSUMSEXCEPTION("Can not assign buffer to runtime tensor view with different shapes!");
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
    RuntimeTensorView<T> &operator OP(const pybind11::buffer & buffer) THROWS(einsums::EinsumsException) {                                 \
        pybind11::gil_scoped_acquire gil;                                                                                                  \
        pybind11::function           override = pybind11::get_override(static_cast<PyTensorView<T> *>(this), #OPNAME);                     \
                                                                                                                                           \
        if (override) {                                                                                                                    \
            auto o = override(buffer);                                                                                                     \
            if (pybind11 ::detail ::cast_is_temporary_value_reference<RuntimeTensorView<T> &>::value) {                                    \
                static pybind11 ::detail ::override_caster_t<RuntimeTensor<T> &> caster;                                                   \
                return pybind11 ::detail ::cast_ref<RuntimeTensorView<T> &>(std ::move(o), caster);                                        \
            }                                                                                                                              \
            return pybind11 ::detail ::cast_safe<RuntimeTensorView<T> &>(std ::move(o));                                                   \
        } else {                                                                                                                           \
            auto buffer_info = buffer.request();                                                                                           \
                                                                                                                                           \
            if (this->rank() != buffer_info.ndim) {                                                                                        \
                throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different rank!");                                 \
            }                                                                                                                              \
            for (int i = buffer_info.ndim - 1; i >= 0; i--) {                                                                              \
                if (this->_dims[i] != buffer_info.shape[i]) {                                                                              \
                    throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different dimensions!");                       \
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

    size_t dim(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, dim, d); }

    std::vector<size_t> dims() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensorView<T>, dims); }

    size_t stride(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, stride, d); }

    std::vector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensorView<T>, strides); }

    bool full_view_of_underlying() const noexcept override { PYBIND11_OVERRIDE(bool, RuntimeTensorView<T>, full_view_of_underlying); }

    const std::string &name() const noexcept override { PYBIND11_OVERRIDE(const std::string &, RuntimeTensorView<T>, name); }

    void set_name(const std::string &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_name, new_name); }

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
        .def("__iter__", [](const PyTensorIterator<T> &copy) { return copy; })
        .def("__reversed__", [](const PyTensorIterator<T> &copy) { return PyTensorIterator<T>(copy, true); });

    auto tensor_view =
        pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>, einsums::detail::RuntimeTensorNoType>(
            mod, ("RuntimeTensorView" + suffix).c_str(), pybind11::buffer_protocol());
    pybind11::class_<RuntimeTensor<T>, PyTensor<T>, SharedRuntimeTensor<T>, einsums::detail::RuntimeTensorNoType>(
        mod, ("RuntimeTensor" + suffix).c_str(), pybind11::buffer_protocol())
        .def(pybind11::init<>())
        .def(pybind11::init<std::string, const std::vector<size_t> &>())
        .def(pybind11::init<const std::vector<size_t> &>())
        .def(pybind11::init<const pybind11::buffer &>())
        .def("zero", &RuntimeTensor<T>::zero)
        .def("set_all", &RuntimeTensor<T>::set_all)
        .def("__getitem__",
             [](RuntimeTensor<T> &self, const pybind11::tuple &args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__getitem__",
             [](RuntimeTensor<T> &self, const pybind11::slice &args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__getitem__",
             [](RuntimeTensor<T> &self, int args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, const pybind11::tuple &key, T value) {
                 PyTensorView<T> cast(self);
                 cast.assign_values(value, key);
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, const pybind11::tuple &key, const pybind11::buffer &values) {
                 PyTensorView<T> cast(self);
                 cast.assign_values(values, key);
             })
#define OPERATOR(OP, TYPE)                                                                                                                 \
    .def(pybind11::self OP TYPE()).def(pybind11::self OP RuntimeTensor<TYPE>()).def(pybind11::self OP RuntimeTensorView<TYPE>())

            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
        .def(pybind11::self *= long())
        .def(
            "__imul__", [](PyTensor<T> &self, const pybind11::buffer &other) -> RuntimeTensor<T> & { return self *= other; },
            pybind11::is_operator()) OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>)
            OPERATOR(/=, std::complex<double>)
        .def(pybind11::self /= long())
        .def(
            "__itruediv__", [](PyTensor<T> &self, const pybind11::buffer &other) -> RuntimeTensor<T> & { return self /= other; },
            pybind11::is_operator()) OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>)
            OPERATOR(+=, std::complex<double>)
        .def(pybind11::self += long())
        .def(
            "__iadd__", [](PyTensor<T> &self, const pybind11::buffer &other) -> RuntimeTensor<T> & { return self += other; },
            pybind11::is_operator()) OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>)
            OPERATOR(-=, std::complex<double>)
        .def(pybind11::self -= long())
        .def(
            "__isub__", [](PyTensor<T> &self, const pybind11::buffer &other) -> RuntimeTensor<T> & { return self -= other; },
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
        .def("__iter__", [](const RuntimeTensor<T> &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](const RuntimeTensor<T> &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensor<T>::rank)
        .def("__copy__", [](const RuntimeTensor<T> &self) { return RuntimeTensor<T>(self); })
        .def("__deepcopy__", [](const RuntimeTensor<T> &self) { return RuntimeTensor<T>(self); })
        .def("copy", [](const RuntimeTensor<T> &self) { return RuntimeTensor<T>(self); })
        .def("deepcopy", [](const RuntimeTensor<T> &self) { return RuntimeTensor<T>(self); })
        .def("__str__",
             [](const RuntimeTensor<T> &self) {
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
        .def(pybind11::init<const RuntimeTensor<T> &>())
        .def(pybind11::init<const RuntimeTensorView<T> &>())
        .def(pybind11::init<RuntimeTensor<T> &, const std::vector<size_t> &>())
        .def(pybind11::init<const RuntimeTensor<T> &, const std::vector<size_t> &>())
        .def(pybind11::init<RuntimeTensorView<T> &, const std::vector<size_t> &>())
        .def(pybind11::init<const RuntimeTensorView<T> &, const std::vector<size_t> &>())
        .def(pybind11::init<pybind11::buffer &>())
        .def("zero", &RuntimeTensorView<T>::zero)
        .def("set_all", &RuntimeTensorView<T>::set_all)
        .def("__getitem__", [](PyTensorView<T> &self, const pybind11::tuple &args) { return self.subscript(args); })
        .def("__getitem__", [](PyTensorView<T> &self, const pybind11::slice &args) { return self.subscript(args); })
        .def("__getitem__", [](PyTensorView<T> &self, int args) { return self.subscript(args); })
        .def("__setitem__", [](PyTensorView<T> &self, const pybind11::tuple &key, T value) { self.assign_values(value, key); })
        .def("__setitem__",
             [](PyTensorView<T> &self, const pybind11::tuple &key, const pybind11::buffer &values) { self.assign_values(values, key); })
            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
        .def(pybind11::self *= long())
        .def(
            "__imul__", [](PyTensorView<T> &self, const pybind11::buffer &other) { return self *= other; }, pybind11::is_operator())
            OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>) OPERATOR(/=, std::complex<double>)
        .def(pybind11::self /= long())
        .def(
            "__itruediv__", [](PyTensorView<T> &self, const pybind11::buffer &other) { return self /= other; }, pybind11::is_operator())
            OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>) OPERATOR(+=, std::complex<double>)
        .def(pybind11::self += long())
        .def(
            "__iadd__", [](PyTensorView<T> &self, const pybind11::buffer &other) { return self += other; }, pybind11::is_operator())
            OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>) OPERATOR(-=, std::complex<double>)
        .def(pybind11::self -= long())
        .def(
            "__isub__", [](PyTensorView<T> &self, const pybind11::buffer &other) { return self -= other; }, pybind11::is_operator())
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
        .def("__iter__", [](const RuntimeTensorView<T> &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](const RuntimeTensorView<T> &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensorView<T>::rank)
        .def("__str__",
             [](const RuntimeTensor<T> &self) {
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

EINSUMS_EXPORT void export_tensor_typeless(pybind11::module_ &mod);

} // namespace einsums::python