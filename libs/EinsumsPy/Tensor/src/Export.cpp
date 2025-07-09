//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>
#include <complex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void export_Tensor(py::module_ &mod) {
    pybind11::class_<einsums::tensor_base::RuntimeTensorNoType, std::shared_ptr<einsums::tensor_base::RuntimeTensorNoType>>(
        mod, "RuntimeTensor");
    pybind11::class_<einsums::tensor_base::RuntimeTensorViewNoType, std::shared_ptr<einsums::tensor_base::RuntimeTensorViewNoType>>(
        mod, "RuntimeTensorView");

    einsums::python::export_tensor<float>(mod);
    einsums::python::export_tensor<double>(mod);
    einsums::python::export_tensor<std::complex<float>>(mod);
    einsums::python::export_tensor<std::complex<double>>(mod);
}
