#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include "einsums/Python.hpp"
#include "einsums/python/PyTesting.hpp"

#include <cstdlib>
#include <cstring>

using namespace einsums::python;
using namespace einsums::python::testing;

BadBuffer::BadBuffer(const BadBuffer &copy)
    : _itemsize{copy._itemsize}, _ndim{copy._ndim}, _dims(copy._dims), _strides(copy._strides), _size{copy._size} {
    if (copy._ptr == nullptr) {
        _ptr = nullptr;

        _size = 0;
    } else {
        _ptr = std::malloc(_size);

        std::memcpy(_ptr, copy._ptr, _size);
    }
}

BadBuffer::BadBuffer(const pybind11::buffer &buffer) {
    auto buffer_info = buffer.request();

    _itemsize = buffer_info.itemsize;
    _ndim     = buffer_info.ndim;
    _dims     = std::vector<size_t>(_ndim);
    _strides  = std::vector<size_t>(_ndim);
    _size     = buffer_info.shape[0] * buffer_info.strides[0];
    _format   = buffer_info.format;

    if (buffer_info.ptr != nullptr) {
        _ptr = std::malloc(_size);

        std::memcpy(_ptr, buffer_info.ptr, _size);
    } else {
        _ptr = nullptr;

        _size = 0;
    }
}

BadBuffer::~BadBuffer() {
    if (_ptr != nullptr) {
        std::free(_ptr);
    }
}

void *BadBuffer::get_ptr() {
    return _ptr;
}

const void *BadBuffer::get_ptr() const {
    return _ptr;
}

void BadBuffer::clear_ptr() {
    if (_ptr != nullptr) {
        std::free(_ptr);

        _size = 0;
    }
}

size_t BadBuffer::get_itemsize() const {
    return _itemsize;
}

void BadBuffer::set_itemsize(size_t size) {
    _itemsize = size;
}

std::string BadBuffer::get_format() const {
    return _format;
}

void BadBuffer::set_format(const std::string &str) {
    _format = str;
}

size_t BadBuffer::get_ndim() const {
    return _ndim;
}

void BadBuffer::set_ndim_noresize(size_t dim) {
    _ndim = dim;
}

void BadBuffer::set_ndim(size_t dim) {
    _ndim = dim;

    _dims.resize(_ndim);
    _strides.resize(_ndim);
}

std::vector<size_t> BadBuffer::get_dims() const {
    return _dims;
}

void BadBuffer::set_dims(const std::vector<size_t> &dims) {
    _dims = dims;
}

void BadBuffer::set_dim(int i, size_t dim) THROWS(std::out_of_range) {
    _dims.at(i) = dim;
}

void BadBuffer::change_dims_size(size_t new_size) {
    _dims.resize(new_size);
}

std::vector<size_t> BadBuffer::get_strides() const {
    return _strides;
}

void BadBuffer::set_strides(const std::vector<size_t> &strides) {
    _strides = strides;
}

void BadBuffer::set_stride(int i, size_t stride) THROWS(std::out_of_range) {
    _strides.at(i) = stride;
}

void BadBuffer::change_strides_size(size_t new_size) {
    _strides.resize(new_size);
}

void einsums::python::export_python_testing(pybind11::module_ &mod) {
    pybind11::class_<BadBuffer>(mod, "BadBuffer", pybind11::buffer_protocol())
        .def(pybind11::init())
        .def(pybind11::init<const BadBuffer &>())
        .def(pybind11::init<const pybind11::buffer &>())
        .def("get_ptr", [](BadBuffer &self) { return self.get_ptr(); })
        .def("clear_ptr", &BadBuffer::clear_ptr)
        .def("get_ndim", &BadBuffer::get_ndim)
        .def("set_ndim", &BadBuffer::set_ndim)
        .def("set_ndim_noresize", &BadBuffer::set_ndim_noresize)
        .def("get_itemsize", &BadBuffer::get_itemsize)
        .def("set_itemsize", &BadBuffer::set_itemsize)
        .def("get_format", &BadBuffer::get_format)
        .def("set_format", &BadBuffer::set_format)
        .def("get_dims", &BadBuffer::get_dims)
        .def("set_dims", &BadBuffer::set_dims)
        .def("set_dim", &BadBuffer::set_dim)
        .def("get_strides", &BadBuffer::get_strides)
        .def("set_strides", &BadBuffer::set_strides)
        .def("set_stride", &BadBuffer::set_stride)
        .def("change_dims_size", &BadBuffer::change_dims_size)
        .def("change_strides_size", &BadBuffer::change_strides_size)
        .def_buffer([](BadBuffer &buf) {
            return pybind11::buffer_info(buf.get_ptr(), buf.get_itemsize(), buf.get_format(), buf.get_ndim(), buf.get_dims(),
                                         buf.get_strides());
        });
}