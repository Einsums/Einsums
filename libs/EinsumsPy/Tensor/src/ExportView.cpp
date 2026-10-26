//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
detail::ExportTensorViewClass<T> export_tensor_view(pybind11::module &mod) {
    std::string suffix = "";

    if constexpr (std::is_same_v<T, float>) {
        suffix = "F";
    } else if constexpr (std::is_same_v<T, double>) {
        suffix = "D";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        suffix = "C";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        suffix = "Z";
    }

    return pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>, einsums::tensor_base::RuntimeTensorNoType>(
        mod, ("RuntimeTensorView" + suffix).c_str(), pybind11::buffer_protocol());
}

void export_tensor_viewf(pybind11::module &mod) {
    auto tensor_view = export_tensor_view<float>(mod);

    export_tensor_view_initf(tensor_view);
    export_tensor_view_opsf(tensor_view);
    export_tensor_view_manipf(tensor_view);
    export_tensor_view_queryf(tensor_view);
}
void export_tensor_viewd(pybind11::module &mod) {
    auto tensor_view = export_tensor_view<double>(mod);

    export_tensor_view_initd(tensor_view);
    export_tensor_view_opsd(tensor_view);
    export_tensor_view_manipd(tensor_view);
    export_tensor_view_queryd(tensor_view);
}
void export_tensor_viewc(pybind11::module &mod) {
    auto tensor_view = export_tensor_view<std::complex<float>>(mod);

    export_tensor_view_initc(tensor_view);
    export_tensor_view_opsc(tensor_view);
    export_tensor_view_manipc(tensor_view);
    export_tensor_view_queryc(tensor_view);
}
void export_tensor_viewz(pybind11::module &mod) {
    auto tensor_view = export_tensor_view<std::complex<double>>(mod);

    export_tensor_view_initz(tensor_view);
    export_tensor_view_opsz(tensor_view);
    export_tensor_view_manipz(tensor_view);
    export_tensor_view_queryz(tensor_view);
}

} // namespace einsums::python