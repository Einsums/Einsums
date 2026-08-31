//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/BlockTensorExport.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>
#include <complex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums::python {

namespace detail {
void fill_with_subscript(pybind11::tuple const &args, std::vector<ptrdiff_t> *out) {
    out->resize(args.size());

    for (int i = 0; i < args.size(); i++) {
        auto const &arg = args[i];

        out->at(i) = pybind11::cast<ptrdiff_t>(arg);
    }
}
} // namespace detail

void export_tensor_typeless(py::module_ &mod) {
    pybind11::class_<einsums::tensor_base::RuntimeTensorNoType, std::shared_ptr<einsums::tensor_base::RuntimeTensorNoType>>(
        mod, "RuntimeTensor");
    pybind11::class_<einsums::tensor_base::RuntimeTensorViewNoType, std::shared_ptr<einsums::tensor_base::RuntimeTensorViewNoType>>(
        mod, "RuntimeTensorView");

    pybind11::class_<einsums::tensor_base::BlockTensorNoExtra, std::shared_ptr<einsums::tensor_base::BlockTensorNoExtra>>(mod,
                                                                                                                          "BlockTensor");

    pybind11::class_<einsums::tensor_base::TiledTensorNoExtra, std::shared_ptr<einsums::tensor_base::TiledTensorNoExtra>>(mod,
                                                                                                                          "TiledTensor");
}

} // namespace einsums::python

void export_Tensor(py::module_ &mod) {
    einsums::python::export_tensor_typeless(mod);

    export_Tensorf(mod);
    export_Tensord(mod);
    export_Tensorc(mod);
    export_Tensorz(mod);
}
