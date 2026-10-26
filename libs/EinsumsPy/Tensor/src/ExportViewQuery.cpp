//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_view_query(detail::ExportTensorViewClass<T> &tensor) {
    tensor.def("dim", &RuntimeTensorView<T>::dim)
        .def("dims", &RuntimeTensorView<T>::dims)
        .def("stride", &RuntimeTensorView<T>::stride)
        .def("strides", &RuntimeTensorView<T>::strides)
        .def("get_name", &RuntimeTensorView<T>::name)
        .def("set_name", &RuntimeTensorView<T>::set_name)
        .def_property("name", &RuntimeTensorView<T>::name, &RuntimeTensorView<T>::set_name)
        .def_property_readonly("shape", [](RuntimeTensorView<T> &self) { return pybind11::cast(self.dims()); })
        .def("size", &RuntimeTensorView<T>::size)
        .def("copy", [](RuntimeTensorView<T> const &self) { return RuntimeTensor<T>(self); })
        .def("is_row_major", [](RuntimeTensorView<T> const &self) -> bool { return self.impl().is_row_major(); })
        .def("is_column_major", [](RuntimeTensorView<T> const &self) -> bool { return self.impl().is_column_major(); })
        .def("__len__", &RuntimeTensorView<T>::size)
        .def("__iter__", [](RuntimeTensorView<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](RuntimeTensorView<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensorView<T>::rank)
        .def("__str__",
             [](RuntimeTensorView<T> const &self) {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def_property_readonly("T", [](RuntimeTensorView<T> const &self) { return detail::transpose(self); })
        .def_buffer([](RuntimeTensorView<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
}

void export_tensor_view_queryf(detail::ExportTensorViewClass<float> &tensor) {
    export_tensor_view_query<float>(tensor);
}

void export_tensor_view_queryd(detail::ExportTensorViewClass<double> &tensor) {
    export_tensor_view_query<double>(tensor);
}

void export_tensor_view_queryc(detail::ExportTensorViewClass<std::complex<float>> &tensor) {
    export_tensor_view_query<std::complex<float>>(tensor);
}

void export_tensor_view_queryz(detail::ExportTensorViewClass<std::complex<double>> &tensor) {
    export_tensor_view_query<std::complex<double>>(tensor);
}

} // namespace einsums::python