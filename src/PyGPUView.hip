#include "einsums/Python.hpp"
#include "einsums/python/PyGPUView.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

using namespace einsums;
using namespace einsums::python;

PyGPUView::PyGPUView(pybind11::buffer &buffer, detail::PyViewMode mode) {
    auto buffer_info = buffer.request(true);

    _host_data = buffer_info.ptr;
    _rank      = buffer_info.ndim;
    _itemsize  = buffer_info.itemsize;
    _fmt_spec  = buffer_info.format;

    _mode = mode;

    if (_mode == detail::COPY) {
        gpu::hip_catch(hipMalloc((void **)&_dev_data, buffer_info.strides[0] * buffer_info.shape[0]));

        gpu::hip_catch(
            hipMemcpy((void *)_dev_data, (const void *)_host_data, buffer_info.strides[0] * buffer_info.shape[0], hipMemcpyHostToDevice));
    } else if (_mode == detail::MAP) {
        gpu::hip_catch(hipHostRegister((void *)_host_data, buffer_info.strides[0] * buffer_info.shape[0], hipHostRegisterDefault));

        gpu::hip_catch(hipHostGetDevicePointer((void **)&_dev_data, (void *)_host_data, 0));
    } else if (_mode == detail::DEVICE_TENSOR) {
        throw EINSUMSEXCEPTION(
            "Can not use DEVICE_TENSOR mode with buffer object. Only used to convert einsums::DeviceTensor to PyGPUView.");
    }

    _alloc_size = buffer_info.strides[0] * buffer_info.shape[0];
    _num_items  = 1;
    for (auto dim : buffer_info.shape) {
        _num_items *= dim;
    }

    _dims.resize(_rank);
    _strides.resize(_rank);

    for (int i = 0; i < _rank; i++) {
        _dims[i]    = buffer_info.shape[i];
        _strides[i] = buffer_info.strides[i];
    }

    gpu::hip_catch(hipMalloc((void **)&_gpu_dims, _rank * sizeof(size_t)));
    gpu::hip_catch(hipMalloc((void **)&_gpu_strides, _rank * sizeof(size_t)));
    gpu::hip_catch(hipMemcpy((void *)_gpu_dims, (const void *)_dims.data(), _rank * sizeof(size_t), hipMemcpyHostToDevice));
    gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), _rank * sizeof(size_t), hipMemcpyHostToDevice));
}

PyGPUView::PyGPUView(const pybind11::buffer &buffer, detail::PyViewMode mode) {
    auto buffer_info = buffer.request(true);

    _host_data = buffer_info.ptr;
    _rank      = buffer_info.ndim;
    _itemsize  = buffer_info.itemsize;
    _fmt_spec  = buffer_info.format;

    _mode = mode;

    if (_mode == detail::COPY) {
        gpu::hip_catch(hipMalloc((void **)&_dev_data, buffer_info.strides[0] * buffer_info.shape[0]));

        gpu::hip_catch(
            hipMemcpy((void *)_dev_data, (const void *)_host_data, buffer_info.strides[0] * buffer_info.shape[0], hipMemcpyHostToDevice));
    } else if (_mode == detail::MAP) {
        gpu::hip_catch(hipHostRegister((void *)_host_data, buffer_info.strides[0] * buffer_info.shape[0], hipHostRegisterDefault));

        gpu::hip_catch(hipHostGetDevicePointer((void **)&_dev_data, (void *)_host_data, 0));
    } else if (_mode == detail::DEVICE_TENSOR) {
        throw EINSUMSEXCEPTION(
            "Can not use DEVICE_TENSOR mode with buffer object. Only used to convert einsums::DeviceTensor to PyGPUView.");
    }

    _alloc_size = buffer_info.strides[0] * buffer_info.shape[0];
    _num_items  = 1;
    for (auto dim : buffer_info.shape) {
        _num_items *= dim;
    }

    _dims.resize(_rank);
    _strides.resize(_rank);

    for (int i = 0; i < _rank; i++) {
        _dims[i]    = buffer_info.shape[i];
        _strides[i] = buffer_info.strides[i];
    }

    gpu::hip_catch(hipMalloc((void **)&_gpu_dims, _rank * sizeof(size_t)));
    gpu::hip_catch(hipMalloc((void **)&_gpu_strides, _rank * sizeof(size_t)));
    gpu::hip_catch(hipMemcpy((void *)_gpu_dims, (const void *)_dims.data(), _rank * sizeof(size_t), hipMemcpyHostToDevice));
    gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), _rank * sizeof(size_t), hipMemcpyHostToDevice));
}

PyGPUView::~PyGPUView() {
    if (_mode == detail::COPY) {
        // Don't copy before. There is no guarantee that the host data is still valid.
        gpu::hip_catch(hipFree(_dev_data));
    } else if (_mode == detail::MAP) {
        gpu::hip_catch(hipHostUnregister((void *)_host_data));
    } else if (_mode == detail::DEVICE_TENSOR) {
        gpu::hip_catch(hipFree(_gpu_strides));
    }

    if (_mode != detail::DEVICE_TENSOR) {
        gpu::hip_catch(hipFree(_gpu_dims));
        gpu::hip_catch(hipFree(_gpu_strides));
    }
}

void *PyGPUView::host_data() {
    return _host_data;
}

const void *PyGPUView::host_data() const {
    return _host_data;
}

void *PyGPUView::dev_data() {
    return _dev_data;
}

const void *PyGPUView::dev_data() const {
    return _dev_data;
}

std::vector<size_t> PyGPUView::dims() const {
    return _dims;
}

size_t PyGPUView::dim(int i) const {
    if (i < 0) {
        return _dims.at(i + _rank);
    }
    return _dims.at(i);
}

std::vector<size_t> PyGPUView::strides() const {
    return _strides;
}

size_t PyGPUView::stride(int i) const {
    if (i < 0) {
        return _strides.at(i + _rank);
    } else {
        return _strides.at(i);
    }
}

std::string PyGPUView::fmt_spec() const {
    return _fmt_spec;
}

size_t *PyGPUView::gpu_dims() {
    return _gpu_dims;
}

const size_t *PyGPUView::gpu_dims() const {
    return _gpu_dims;
}

size_t *PyGPUView::gpu_strides() {
    return _gpu_strides;
}

const size_t *PyGPUView::gpu_strides() const {
    return _gpu_strides;
}

size_t PyGPUView::rank() const {
    return _rank;
}

size_t PyGPUView::size() const {
    return _num_items;
}

size_t PyGPUView::itemsize() const {
    return _itemsize;
}

void PyGPUView::update_H2D() {
    if(_mode == detail::COPY) {
        gpu::hip_catch(hipMemcpy(_dev_data, _host_data, _alloc_size, hipMemcpyHostToDevice));
    }
}

void PyGPUView::update_D2H() {
    if(_mode == detail::COPY) {
        gpu::hip_catch(hipMemcpy(_host_data, _dev_data, _alloc_size, hipMemcpyDeviceToHost));
    }
}

void einsums::python::export_gpu_view(pybind11::module_ &mod) {
    pybind11::enum_<detail::PyViewMode>(mod, "GPUViewMode")
        .value("COPY", detail::COPY)
        .value("MAP", detail::MAP)
        .value("DEVICE_TENSOR", detail::DEVICE_TENSOR)
        .export_values();

    pybind11::class_<PyGPUView>(mod, "GPUView")
        .def(pybind11::init<pybind11::buffer &, detail::PyViewMode>())
        .def(pybind11::init<const pybind11::buffer &, detail::PyViewMode>())
        .def("dims", &PyGPUView::dims)
        .def("strides", &PyGPUView::strides)
        .def("dim", &PyGPUView::dim)
        .def("stride", &PyGPUView::stride)
        .def("fmt_spec", &PyGPUView::fmt_spec)
        .def("update_H2D", &PyGPUView::update_H2D)
        .def("update_D2H", &PyGPUView::update_D2H)
        .def("size", &PyGPUView::size)
        .def("__len__", &PyGPUView::size)
        .def("rank", &PyGPUView::rank)
        .def("itemsize", &PyGPUView::itemsize);
}