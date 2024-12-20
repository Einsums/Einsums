#include <EinsumsPy/TestUtils/Export.hpp>
#include <EinsumsPy/TestUtils/BadBuffer.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace einsums::python::testing;


void export_TestUtils(py::module_ &mod) {
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