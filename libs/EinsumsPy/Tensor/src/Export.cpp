//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BlockManager/BlockManager.hpp>

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

#ifdef EINSUMS_COMPUTE_CODE
    pybind11::class_<einsums::GPUBlock, std::shared_ptr<einsums::GPUBlock>>(mod, "GPULock");
#endif

    einsums::python::export_tensor<float>(mod);
    einsums::python::export_tensor<double>(mod);
    einsums::python::export_tensor<std::complex<float>>(mod);
    einsums::python::export_tensor<std::complex<double>>(mod);
}
