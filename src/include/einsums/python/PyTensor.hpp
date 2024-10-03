#pragma once

#include "einsums/python/RuntimeTensor.hpp"
#include "einsums/utility/IndexUtils.hpp"

#include <pybind11/attr.h>
#include <pybind11/detail/common.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace einsums::python {

// Forward declaration
template <typename T>
class PyTensorView;

template <typename T>
class PyTensorIterator {
  private:
    std::mutex           _lock;
    size_t               _curr_index, _elements;
    std::vector<size_t>  _index_strides;
    RuntimeTensorView<T> _tensor;
    bool                 _stop{false}, _reverse{false};

  public:
    PyTensorIterator(const RuntimeTensor<T> &other, bool reverse = false)
        : _tensor{other}, _reverse{reverse}, _index_strides(other.rank()), _elements(other.size()) {
        if (!reverse) {
            _curr_index = 0;
        } else {
            _curr_index = other.size();
        }

        tensor_algebra::detail::dims_to_strides(other.dims(), _index_strides);
    }

    PyTensorIterator(const RuntimeTensorView<T> &other, bool reverse = false)
        : _tensor{other}, _reverse{reverse}, _index_strides(other.rank()), _elements(other.size()) {
        if (!reverse) {
            _curr_index = 0;
        } else {
            _curr_index = other.size();
        }

        tensor_algebra::detail::dims_to_strides(other.dims(), _index_strides);
    }

    T next() {
        _lock.lock();

        if (_stop) {
            _lock.unlock();
            throw pybind11::stop_iteration();
        }

        std::vector<size_t> ind;

        tensor_algebra::detail::sentinel_to_indices(_curr_index, _index_strides, ind);

        T &out = _tensor(ind);

        if (_reverse) {
            _curr_index--;
        } else {
            _curr_index++;
        }

        if (_curr_index < 0 || _curr_index >= _elements) {
            _stop = true;
        }

        _lock.unlock();

        return out;
    }

    bool reversed() const { return _reverse; }
};

template <typename T>
class PyTensor : RuntimeTensor<T> {
  public:
    using RuntimeTensor<T>::RuntimeTensor;

    PyTensor(const PyTensor<T> &) = default;
    PyTensor(const std::shared_ptr<PyTensor<T>> &other) : PyTensor(*other) {}

    virtual ~PyTensor() = default;

    void zero() override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, zero); }

    void set_all(T val) override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, set_all, val); }

    pybind11::object subscript(const pybind11::tuple &args) override {
        PYBIND11_OVERRIDE_NAME(pybind11::object, RuntimeTensor<T>, "__subscript", subscript, args);
    }

    const pybind11::object subscript(const pybind11::tuple &args) const override {
        PYBIND11_OVERRIDE_NAME(const pybind11::object, RuntimeTensor<T>, "__subscript", subscript, args);
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, ptrdiff_t index) override {
        PYBIND11_OVERRIDE_NAME(RuntimeTensorView<T>, RuntimeTensor<T>, "__assign", assign_values, value, index);
    }

    pybind11::object assign_values(T value, ptrdiff_t index) override {
        PYBIND11_OVERRIDE_NAME(pybind11::object, RuntimeTensor<T>, "__assign", assign_values, value, index);
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, const pybind11::slice &index) override {
        PYBIND11_OVERRIDE_NAME(RuntimeTensorView<T>, RuntimeTensor<T>, "__assign", assign_values, value, index);
    }

    RuntimeTensorView<T> assign_values(T value, const pybind11::slice &index) override {
        PYBIND11_OVERRIDE_NAME(RuntimeTensorView<T>, RuntimeTensor<T>, "__assign", assign_values, value, index);
    }

    RuntimeTensor<T> &operator=(const RuntimeTensor<T> &other) override {
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, operator=, other);
    }

    RuntimeTensor<T> &operator=(const RuntimeTensorView<T> &other) override {
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, operator=, other);
    }

    RuntimeTensor<T> &operator=(T other) override { PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, operator=, other); }

    RuntimeTensor<T> &operator=(const pybind11::buffer &buffer) override {
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, operator=, buffer);
    }

#define OPERATOR(OP)                                                                                                                       \
    RuntimeTensor<T> &OP(const T &other) override { PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OP, other); }                  \
    RuntimeTensor<T> &OP(const RuntimeTensor<T> &other) override { PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OP, other); }   \
    RuntimeTensor<T> &OP(const RuntimeTensorView<T> &other) override {                                                                     \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OP, other);                                                                \
    }                                                                                                                                      \
    RuntimeTensor<T> &OP(const pybind11::buffer &buffer) override { PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, OP, buffer); }

    OPERATOR(operator*=)
    OPERATOR(operator/=)
    OPERATOR(operator+=)
    OPERATOR(operator-=)

#undef OPERATOR

    size_t dim(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, dim, d); }

    std::vector<size_t> dims() const override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensor<T>, dims); }

    const typename RuntimeTensor<T>::Vector &vector_data() const override {
        PYBIND11_OVERRIDE(const typename RuntimeTensor<T>::Vector &, RuntimeTensor<T>, vector_data);
    }

    typename RuntimeTensor<T>::Vector &vector_data() override {
        PYBIND11_OVERRIDE(typename RuntimeTensor<T>::Vector &, RuntimeTensor<T>, vector_data);
    }

    size_t stride(int d) const noexcept override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, stride, d); }

    std::vector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensor<T>, strides); }

    RuntimeTensorView<T> to_rank_1_view() const override { PYBIND11_OVERRIDE(RuntimeTensorView<T>, RuntimeTensor<T>, to_rank_1_view); }

    bool full_view_of_underlying() const noexcept override { PYBIND11_OVERRIDE(bool, RuntimeTensor<T>, full_view_of_underlying); }

    const std::string &name() const override { PYBIND11_OVERRIDE(const std::string &, RuntimeTensor<T>, name); }

    void set_name(const std::string &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, set_name, new_name); }

    size_t rank() const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, rank); }
}; // namespace einsums::python

template <typename T>
class PyTensorView : RuntimeTensorView<T> {
  public:
    using RuntimeTensorView<T>::RuntimeTensorView;

    PyTensorView(const PyTensorView<T> &) = default;
    PyTensorView(const RuntimeTensorView<T> &copy) : RuntimeTensorView<T>(copy) {}

    virtual ~PyTensorView() = default;

    void zero() override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, zero); }

    void set_all(T val) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_all, val); }

    pybind11::object subscript(const pybind11::tuple &args) override {
        PYBIND11_OVERRIDE_NAME(pybind11::object, RuntimeTensorView<T>, "__subscript", subscript, args);
    }

    const pybind11::object subscript(const pybind11::tuple &args) const override {
        PYBIND11_OVERRIDE_NAME(const pybind11::object, RuntimeTensorView<T>, "__subscript", subscript, args);
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, int index) override {
        PYBIND11_OVERRIDE_NAME(RuntimeTensorView<T>, RuntimeTensorView<T>, "__assign", assign_values, value, index);
    }

    pybind11::object assign_values(T value, int index) override {
        PYBIND11_OVERRIDE_NAME(pybind11::object, RuntimeTensorView<T>, "__assign", assign_values, value, index);
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, const pybind11::slice &index) override {
        PYBIND11_OVERRIDE_NAME(RuntimeTensorView<T>, RuntimeTensorView<T>, "__assign", assign_values, value, index);
    }

    RuntimeTensorView<T> assign_values(T value, const pybind11::slice &index) override {
        PYBIND11_OVERRIDE_NAME(RuntimeTensorView<T>, RuntimeTensorView<T>, "__assign", assign_values, value, index);
    }

    RuntimeTensorView<T> &operator=(const RuntimeTensor<T> &other) override {
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, operator=, other);
    }

    RuntimeTensorView<T> &operator=(const RuntimeTensorView<T> &other) override {
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, operator=, other);
    }

    RuntimeTensorView<T> &operator=(T other) override { PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, operator=, other); }

#define OPERATOR(OP)                                                                                                                       \
    RuntimeTensorView<T> &OP(const T &other) override { PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OP, other); }      \
    RuntimeTensorView<T> &OP(const RuntimeTensor<T> &other) override {                                                                     \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OP, other);                                                        \
    }                                                                                                                                      \
    RuntimeTensorView<T> &OP(const RuntimeTensorView<T> &other) override {                                                                 \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OP, other);                                                        \
    }                                                                                                                                      \
    RuntimeTensorView<T> &OP(const pybind11::buffer &other) override {                                                                     \
        PYBIND11_OVERRIDE(RuntimeTensorView<T> &, RuntimeTensorView<T>, OP, other);                                                        \
    }

    OPERATOR(operator*=)
    OPERATOR(operator/=)
    OPERATOR(operator+=)
    OPERATOR(operator-=)

#undef OPERATOR

    size_t dim(int d) const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, dim, d); }

    std::vector<size_t> dims() const override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensorView<T>, dims); }

    size_t stride(int d) const noexcept override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, stride, d); }

    std::vector<size_t> strides() const noexcept override { PYBIND11_OVERRIDE(std::vector<size_t>, RuntimeTensorView<T>, strides); }

    bool full_view_of_underlying() const noexcept override { PYBIND11_OVERRIDE(bool, RuntimeTensorView<T>, full_view_of_underlying); }

    const std::string &name() const override { PYBIND11_OVERRIDE(const std::string &, RuntimeTensorView<T>, name); }

    void set_name(const std::string &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_name, new_name); }

    size_t rank() const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, rank); }
};

template <typename T>
void export_tensor(pybind11::module &mod) {
    std::string suffix = "";

    if constexpr (std::is_same_v<T, float>) {
        suffix = "F";
    } else if constexpr (std::is_same_v<T, double>) {
        suffix = "";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        suffix = "C";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        suffix = "Z";
    }

    pybind11::class_<PyTensorIterator<T>, std::shared_ptr<PyTensorIterator<T>>>(mod, ("RuntimeTensorIterator" + suffix).c_str())
        .def("__next__", &PyTensorIterator<T>::next, pybind11::return_value_policy::reference)
        .def("reversed", &PyTensorIterator<T>::reversed);

    auto tensor_view = pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>>(
        mod, ("RuntimeTensorView" + suffix).c_str(), pybind11::buffer_protocol());
    pybind11::class_<RuntimeTensor<T>, PyTensor<T>, SharedRuntimeTensor<T>>(mod, ("RuntimeTensor" + suffix).c_str(),
                                                                            pybind11::buffer_protocol())
        .def(pybind11::init<>())
        .def(pybind11::init<std::string, const std::vector<size_t> &>())
        .def(pybind11::init<const std::vector<size_t> &>())
        .def("zero", &RuntimeTensor<T>::zero)
        .def("set_all", &RuntimeTensor<T>::set_all)
        .def("__getitem__", [](const RuntimeTensor<T> &self, const pybind11::tuple &args) { return self.subscript(args); })
        .def("__getitem__", [](const RuntimeTensor<T> &self, const pybind11::slice &args) { return self.subscript(args); })
        .def("__getitem__", [](const RuntimeTensor<T> &self, int args) { return self.subscript(args); })
        .def("__setitem__", [](RuntimeTensor<T> &self, const pybind11::tuple &key, T value) { self.assign_values(value, key); })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, const pybind11::tuple &key, const pybind11::buffer &values) { self.assign_values(values, key); })
        .def(pybind11::self *= T())
        .def(pybind11::self *= pybind11::self)
        .def(pybind11::self *= RuntimeTensorView<T>())
        .def(pybind11::self *= pybind11::buffer())
        .def(pybind11::self /= T())
        .def(pybind11::self /= pybind11::self)
        .def(pybind11::self /= RuntimeTensorView<T>())
        .def(pybind11::self /= pybind11::buffer())
        .def(pybind11::self += T())
        .def(pybind11::self += pybind11::self)
        .def(pybind11::self += RuntimeTensorView<T>())
        .def(pybind11::self += pybind11::buffer())
        .def(pybind11::self -= T())
        .def(pybind11::self -= pybind11::self)
        .def(pybind11::self -= RuntimeTensorView<T>())
        .def(pybind11::self -= pybind11::buffer())
        .def("assign", [](RuntimeTensor<T> &self, pybind11::buffer &buffer) { self = buffer; })
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
        .def("__getitem__", [](const RuntimeTensorView<T> &self, const pybind11::tuple &args) { return self.subscript(args); })
        .def("__getitem__", [](const RuntimeTensorView<T> &self, const pybind11::slice &args) { return self.subscript(args); })
        .def("__getitem__", [](const RuntimeTensorView<T> &self, int args) { return self.subscript(args); })
        .def("__setitem__", [](RuntimeTensorView<T> &self, const pybind11::tuple &key, T value) { self.assign_values(value, key); })
        .def("__setitem__", [](RuntimeTensorView<T> &self, const pybind11::tuple &key,
                               const pybind11::buffer &values) { self.assign_values(values, key); })
        .def(pybind11::self *= T())
        .def(pybind11::self *= pybind11::self)
        .def(pybind11::self *= RuntimeTensor<T>())
        .def(pybind11::self *= pybind11::buffer())
        .def(pybind11::self /= T())
        .def(pybind11::self /= pybind11::self)
        .def(pybind11::self /= RuntimeTensor<T>())
        .def(pybind11::self /= pybind11::buffer())
        .def(pybind11::self += T())
        .def(pybind11::self += pybind11::self)
        .def(pybind11::self += RuntimeTensor<T>())
        .def(pybind11::self += pybind11::buffer())
        .def(pybind11::self -= T())
        .def(pybind11::self -= pybind11::self)
        .def(pybind11::self -= RuntimeTensor<T>())
        .def(pybind11::self -= pybind11::buffer())
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
        .def_buffer([](RuntimeTensorView<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
}

} // namespace einsums::python