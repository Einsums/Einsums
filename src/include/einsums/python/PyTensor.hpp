#pragma once

#include "einsums/python/RuntimeTensor.hpp"

#include <pybind11/attr.h>
#include <pybind11/pybind11.h>

namespace einsums::python {

// Forward declaration
template <typename T>
class PyTensorView;

template <typename T>
class PyTensorIterator {
  private:
    std::mutex            _lock;
    std::vector<ssize_t>  _curr_index;
    RuntimeTensorView<T> &_tensor;
    bool                  _stop{false}, _reverse{false};

  public:
    PyTensorIterator(RuntimeTensor<T> &other, bool reverse = false) : _curr_index(other.rank()), _tensor{other}, _reverse{reverse} {
        for (int i = 0; i < _curr_index.size(); i++) {
            _curr_index[i] = 0;
        }
    }

    PyTensorIterator(const RuntimeTensor<T> &other, bool reverse = false) : _curr_index(other.rank()), _tensor{other}, _reverse{reverse} {
        for (int i = 0; i < _curr_index.size(); i++) {
            _curr_index[i] = 0;
        }
    }

    PyTensorIterator(RuntimeTensorView<T> &other, bool reverse = false) : _curr_index(other.rank()), _tensor{other}, _reverse{reverse} {
        for (int i = 0; i < _curr_index.size(); i++) {
            _curr_index[i] = 0;
        }
    }

    PyTensorIterator(const RuntimeTensorView<T> &other, bool reverse = false)
        : _curr_index(other.rank()), _tensor{other}, _reverse{reverse} {
        for (int i = 0; i < _curr_index.size(); i++) {
            _curr_index[i] = 0;
        }
    }

    T &next() {
        _lock.lock();

        if (_stop) {
            _lock.unlock();
            throw pybind11::stop_iteration();
        }

        T &out = _tensor(_curr_index);

        if (!_reverse) {
            _curr_index[0] += 1;

            for (int i = 0; i < _curr_index.size() - 1 && _curr_index[i] >= _tensor.dim(i); i++) {
                _curr_index[i] = 0;
                _curr_index[i + 1] += 1;
            }

            if (_curr_index[_curr_index.size() - 1] >= _tensor.dim(_curr_index.size() - 1)) {
                _stop = true;
            }
        } else {
            _curr_index[0] -= 1;

            for (int i = 0; i < _curr_index.size() - 1 && _curr_index[i] < 0; i++) {
                _curr_index[i] = _tensor.dim(i) - 1;
                _curr_index[i + 1] -= 1;
            }

            if (_curr_index[_curr_index.size() - 1] < 0) {
                _stop = true;
            }
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

    T &set_value_at(T value, const std::vector<size_t> &index) override {
        PYBIND11_OVERRIDE_NAME(T &, RuntimeTensor<T>, "__set_value_at", set_value_at, value, index);
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

    bool full_view_of_underlying() const override { PYBIND11_OVERRIDE(bool, RuntimeTensor<T>, full_view_of_underlying); }

    const std::string &name() const override { PYBIND11_OVERRIDE(const std::string &, RuntimeTensor<T>, name); }

    void set_name(const std::string &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensor<T>, set_name, new_name); }

    size_t rank() const override { PYBIND11_OVERRIDE(size_t, RuntimeTensor<T>, rank); }
}; // namespace einsums::python

template <typename T>
class PyTensorView : RuntimeTensorView<T> {
  public:
    using RuntimeTensorView<T>::RuntimeTensorView;

    virtual ~PyTensorView() = default;

    void zero() override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, zero); }

    void set_all(T val) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_all, val); }

    T &subscript(const pybind11::tuple &args) override {
        PYBIND11_OVERRIDE_NAME(T &, RuntimeTensorView<T>, "__subscript", subscript, args);
    }

    const T &subscript(const pybind11::tuple &args) const override {
        PYBIND11_OVERRIDE_NAME(const T &, RuntimeTensorView<T>, "__subscript", subscript, args);
    }

    T &set_value_at(T value, const std::vector<size_t> &index) override {
        PYBIND11_OVERRIDE_NAME(T &, RuntimeTensorView<T>, "__set_value_at", set_value_at, value, index);
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

    bool full_view_of_underlying() const override { PYBIND11_OVERRIDE(bool, RuntimeTensorView<T>, full_view_of_underlying); }

    const std::string &name() const override { PYBIND11_OVERRIDE(const std::string &, RuntimeTensorView<T>, name); }

    void set_name(const std::string &new_name) override { PYBIND11_OVERRIDE(void, RuntimeTensorView<T>, set_name, new_name); }

    size_t rank() const override { PYBIND11_OVERRIDE(size_t, RuntimeTensorView<T>, rank); }
};

template <typename T>
void export_tensor(pybind11::module &mod) {
    pybind11::class_<PyTensorIterator<T>, std::shared_ptr<PyTensorIterator<T>>>(mod, "RuntimeTensorIterator")
        .def("__next__", &PyTensorIterator<T>::next)
        .def("reversed", &PyTensorIterator<T>::reversed);
    auto tensor_view = pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>>(mod, "RuntimeTensorView",
                                                                                                           pybind11::buffer_protocol());
    pybind11::class_<RuntimeTensor<T>, PyTensor<T>, SharedRuntimeTensor<T>>(mod, "RuntimeTensor", pybind11::buffer_protocol())
        .def(pybind11::init<>())
        .def(pybind11::init<std::string, const std::vector<size_t> &>())
        .def(pybind11::init<const std::vector<size_t> &>())
        .def("zero", &RuntimeTensor<T>::zero)
        .def("set_all", &RuntimeTensor<T>::set_all)
        .def("__getitem__", &RuntimeTensor<T>::subscript)
        .def("__setitem__", [](RuntimeTensor<T> &self, const pybind11::tuple &key, T value) { self.assign_values(value, key); })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, const pybind11::tuple &key, const pybind11::buffer &values) { self.assign_values(values, key); })
        .def("__imult__", &RuntimeTensor<T>::operator*=, pybind11::is_operator())
        .def("__idiv__", &RuntimeTensor<T>::operator/=, pybind11::is_operator())
        .def("__iadd__", &RuntimeTensor<T>::operator+=, pybind11::is_operator())
        .def("__isub__", &RuntimeTensor<T>::operator-=, pybind11::is_operator())
        .def("dim", &RuntimeTensor<T>::dim)
        .def("dims", &RuntimeTensor<T>::dims)
        .def("stride", &RuntimeTensor<T>::stride)
        .def("strides", &RuntimeTensor<T>::strides)
        .def("vector_data", &RuntimeTensor<T>::vector_data)
        .def("to_rank_1_view", &RuntimeTensor<T>::to_rank_1_view)
        .def("get_name", &RuntimeTensor<T>::name)
        .def("set_name", &RuntimeTensor<T>::set_name)
        .def_property("name", &RuntimeTensor<T>::name, &RuntimeTensor<T>::set_name)
        .def("size", &RuntimeTensor<T>::size)
        .def("__len__", &RuntimeTensor<T>::size)
        .def("__iter__", [](const RuntimeTensor<T> &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](const RuntimeTensor<T> &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensor<T>::rank)
        .def_buffer([](RuntimeTensor<T> &self) {
            std::vector<ssize_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dims(i);
                strides[i] = sizeof(T) * self.strides(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
    tensor_view.def(pybind11::init<>())
        .def(pybind11::init<std::string, const std::vector<size_t> &>())
        .def(pybind11::init<const std::vector<size_t> &>())
        .def("zero", &RuntimeTensorView<T>::zero)
        .def("set_all", &RuntimeTensorView<T>::set_all)
        .def("__getitem__", &RuntimeTensorView<T>::subscript)
        .def("__setitem__", [](RuntimeTensorView<T> &self, const pybind11::tuple &key, T value) { self.assign_values(value, key); })
        .def("__setitem__", [](RuntimeTensorView<T> &self, const pybind11::tuple &key,
                               const pybind11::buffer &values) { self.assign_values(values, key); })
        .def("__imult__", &RuntimeTensorView<T>::operator*=, pybind11::is_operator())
        .def("__idiv__", &RuntimeTensorView<T>::operator/=, pybind11::is_operator())
        .def("__iadd__", &RuntimeTensorView<T>::operator+=, pybind11::is_operator())
        .def("__isub__", &RuntimeTensorView<T>::operator-=, pybind11::is_operator())
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
            std::vector<ssize_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dims(i);
                strides[i] = sizeof(T) * self.strides(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
}

} // namespace einsums::python