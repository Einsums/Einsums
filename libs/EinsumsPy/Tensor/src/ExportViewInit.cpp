//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_view_init(detail::ExportTensorViewClass<T> &tensor) {
    tensor.def(pybind11::init<>())
        .def(pybind11::init<RuntimeTensor<T> &>())
        .def(pybind11::init<RuntimeTensor<T> const &>())
        .def(pybind11::init<RuntimeTensorView<T> const &>())
        .def(pybind11::init<RuntimeTensor<T> &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensor<T> const &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensorView<T> &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensorView<T> const &, std::vector<size_t> const &>())
        .def(pybind11::init<pybind11::buffer &>());
}

void export_tensor_view_initf(detail::ExportTensorViewClass<float> &tensor) {
    export_tensor_view_init<float>(tensor);
}

void export_tensor_view_initd(detail::ExportTensorViewClass<double> &tensor) {
    export_tensor_view_init<double>(tensor);
}

void export_tensor_view_initc(detail::ExportTensorViewClass<std::complex<float>> &tensor) {
    export_tensor_view_init<std::complex<float>>(tensor);
}

void export_tensor_view_initz(detail::ExportTensorViewClass<std::complex<double>> &tensor) {
    export_tensor_view_init<std::complex<double>>(tensor);
}

} // namespace einsums::python