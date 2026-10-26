//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_init(detail::ExportTensorClass<T> &tensor) {
    tensor.def(pybind11::init<>())
        .def(pybind11::init<std::string, std::vector<size_t> const &>())
        .def(pybind11::init<std::vector<size_t> const &>())
        .def(pybind11::init<pybind11::buffer const &>());
}

void export_tensor_initf(detail::ExportTensorClass<float> &tensor) {
    export_tensor_init<float>(tensor);
}

void export_tensor_initd(detail::ExportTensorClass<double> &tensor) {
    export_tensor_init<double>(tensor);
}

void export_tensor_initc(detail::ExportTensorClass<std::complex<float>> &tensor) {
    export_tensor_init<std::complex<float>>(tensor);
}

void export_tensor_initz(detail::ExportTensorClass<std::complex<double>> &tensor) {
    export_tensor_init<std::complex<double>>(tensor);
}

} // namespace einsums::python