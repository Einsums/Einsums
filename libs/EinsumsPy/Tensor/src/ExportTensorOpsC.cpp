//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>
#include "ExportTensorOps.hpp"

namespace einsums::python {


void export_tensor_opsc(detail::ExportTensorClass<std::complex<float>> &tensor) {
    export_tensor_ops<std::complex<float>>(tensor);
}

} // namespace einsums::python