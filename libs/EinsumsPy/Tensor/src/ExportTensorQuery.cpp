//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_query(detail::ExportTensorClass<T> &tensor) {
    tensor.def("dim", &RuntimeTensor<T>::dim)
        .def("dims", &RuntimeTensor<T>::dims)
        .def("stride", &RuntimeTensor<T>::stride)
        .def("strides", &RuntimeTensor<T>::strides)
        .def("to_rank_1_view", &RuntimeTensor<T>::to_rank_1_view)
        .def("get_name", &RuntimeTensor<T>::name)
        .def("set_name", &RuntimeTensor<T>::set_name)
        .def_property("name", &RuntimeTensor<T>::name, &RuntimeTensor<T>::set_name)
        .def_property_readonly("shape", [](RuntimeTensor<T> &self) { return pybind11::cast(self.dims()); })
        .def("size", &RuntimeTensor<T>::size)
        .def("is_row_major", [](RuntimeTensor<T> const &self) -> bool { return self.impl().is_row_major(); })
        .def("is_column_major", [](RuntimeTensor<T> const &self) -> bool { return self.impl().is_column_major(); })
        .def("__len__", &RuntimeTensor<T>::size)
        .def("__iter__", [](RuntimeTensor<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](RuntimeTensor<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensor<T>::rank)
        .def("__copy__", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__deepcopy__", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("copy", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("deepcopy", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__str__",
             [](RuntimeTensor<T> const &self) {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def_property_readonly("T", [](RuntimeTensor<T> const &self) { return detail::transpose(self); })
        .def_buffer([](RuntimeTensor<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
}

void export_tensor_queryf(detail::ExportTensorClass<float> &tensor) {
    export_tensor_query<float>(tensor);
}

void export_tensor_queryd(detail::ExportTensorClass<double> &tensor) {
    export_tensor_query<double>(tensor);
}

void export_tensor_queryc(detail::ExportTensorClass<std::complex<float>> &tensor) {
    export_tensor_query<std::complex<float>>(tensor);
}

void export_tensor_queryz(detail::ExportTensorClass<std::complex<double>> &tensor) {
    export_tensor_query<std::complex<double>>(tensor);
}

} // namespace einsums::python