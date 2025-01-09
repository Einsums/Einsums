//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <EinsumsPy/GPU/PyGPUView.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace einsums::python;

void throw_hip(int status) {
    einsums::hip_catch(static_cast<hipError_t>(status));
}

void throw_hipblas(int status) {
    einsums::hipblas_catch(static_cast<hipblasStatus_t>(status));
}

void throw_hipsolver(int status) {
    einsums::hipsolver_catch(static_cast<hipsolverStatus_t>(status));
}

void export_GPU(py::module_ &mod) {
    pybind11::enum_<detail::PyViewMode>(mod, "GPUViewMode")
        .value("COPY", detail::COPY)
        .value("MAP", detail::MAP)
        .value("DEVICE_TENSOR", detail::DEVICE_TENSOR)
        .export_values();

    pybind11::class_<PyGPUView>(mod, "GPUView")
        .def(pybind11::init<pybind11::buffer &, detail::PyViewMode>())
        .def("dims", &PyGPUView::dims)
        .def("strides", &PyGPUView::strides)
        .def("dim", &PyGPUView::dim)
        .def("stride", &PyGPUView::stride)
        .def("fmt_spec", &PyGPUView::fmt_spec)
        .def("update_H2D", &PyGPUView::update_H2D)
        .def("update_D2H", &PyGPUView::update_D2H)
        .def("size", &PyGPUView::size)
        .def("__len__", &PyGPUView::size)
        .def("rank", &PyGPUView::Rank)
        .def("itemsize", &PyGPUView::itemsize);
    
    mod.def("throw_hip", throw_hip)
    .def("throw_hipblas", throw_hipblas)
    .def("throw_hipsolver", throw_hipsolver);
}