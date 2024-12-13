#include <Einsums/Config.hpp>

#include <EinsumsPy/TestUtils/BadBuffer.hpp>
#include <pybind11/pybind11.h>

using namespace einsums::python;
using namespace einsums::python::testing;

BadBuffer::BadBuffer(BadBuffer const &copy)
    : _itemsize{copy._itemsize}, _ndim{copy._ndim}, _dims(copy._dims), _strides(copy._strides), _size{copy._size} {
    if (copy._ptr == nullptr) {
        _ptr = nullptr;

        _size = 0;
    } else {
        _ptr = std::malloc(_size);

        std::memcpy(_ptr, copy._ptr, _size);
    }
}

BadBuffer::BadBuffer(pybind11::buffer const &buffer) {
    auto buffer_info = buffer.request();

    _itemsize = buffer_info.itemsize;
    _ndim     = buffer_info.ndim;
    _dims     = std::vector<size_t>(buffer_info.shape.size());
    _strides  = std::vector<size_t>(buffer_info.strides.size());
    _size     = buffer_info.shape[0] * buffer_info.strides[0];
    _format   = buffer_info.format;

    for(int i = 0; i < _dims.size(); i++) {
        _dims[i] = buffer_info.shape[i];
    }

    for(int i = 0; i < _strides.size(); i++) {
        _strides[i] = buffer_info.strides[i];
    }

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

void const *BadBuffer::get_ptr() const {
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

void BadBuffer::set_format(std::string const &str) {
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

void BadBuffer::set_dims(std::vector<size_t> const &dims) {
    _dims = dims;
}

void BadBuffer::set_dim(int i, size_t dim) {
    _dims.at(i) = dim;
}

void BadBuffer::change_dims_size(size_t new_size) {
    _dims.resize(new_size);
}

std::vector<size_t> BadBuffer::get_strides() const {
    return _strides;
}

void BadBuffer::set_strides(std::vector<size_t> const &strides) {
    _strides = strides;
}

void BadBuffer::set_stride(int i, size_t stride) {
    _strides.at(i) = stride;
}

void BadBuffer::change_strides_size(size_t new_size) {
    _strides.resize(new_size);
}
