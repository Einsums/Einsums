#include <EinsumsPy/GPU/Export.hpp>
#include <EinsumsPy/GPU/PyGPUView.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace einsums::python;

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
        .def("rank", &PyGPUView::rank)
        .def("itemsize", &PyGPUView::itemsize);
}