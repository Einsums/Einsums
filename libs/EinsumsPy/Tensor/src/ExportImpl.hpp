//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// No pragma once

#include <EinsumsPy/Tensor/BlockTensorExport.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/PyTiledTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>
#include <complex>
#include <pybind11/pybind11.h>

#ifndef EXPORT_FUNCTION
#    define EXPORT_FUNCTION2(name, suffix) name##suffix
#    define EXPORT_FUNCTION(name, suffix)  EXPORT_FUNCTION2(name, suffix)
#endif

namespace py = pybind11;

void EXPORT_FUNCTION(export_Tensor, EINSUMS_EXPORT_TYPE_SUFFIX)(py::module_ &mod) {
    einsums::python::export_tensor<EINSUMS_EXPORT_TYPE>(mod);

    einsums::python::export_block_tensor<EINSUMS_EXPORT_TYPE>(mod);

    einsums::python::export_tiled_tensor<EINSUMS_EXPORT_TYPE>(mod);
}
