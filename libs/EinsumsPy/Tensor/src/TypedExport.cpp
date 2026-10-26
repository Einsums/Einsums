//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>
#include <complex>

namespace einsums::python {
/**
 * @brief Expose runtime tensors to Python.
 *
 * @tparam T The stored type of the tensors to export.
 * @param mod The module which will contain the definitions.
 */
template <typename T>
detail::ExportTensorClass<T> export_tensor(pybind11::module &mod) {
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

    mod.def(("create_random_tensor" + suffix).c_str(),
            [](std::string const &name, std::vector<size_t> const &dims) { return einsums::create_random_tensor<T>(name, dims); })
        .def(("create_random_definite" + suffix).c_str(),
             [](std::string const &name, size_t rows, RemoveComplexT<T> mean) {
                 return RuntimeTensor<T>(einsums::create_random_definite<T>(name, rows, rows));
             },
             pybind11::arg("name"), pybind11::arg("rows"), pybind11::arg("mean") = RemoveComplexT<T>{1.0})
        .def(("create_random_semidefinite" + suffix).c_str(),
             [](std::string const &name, size_t rows, RemoveComplexT<T> mean, int force_zeros) {
                 return RuntimeTensor<T>(einsums::create_random_semidefinite<T>(name, rows, rows));
             },
             pybind11::arg("name"), pybind11::arg("rows"), pybind11::arg("mean") = RemoveComplexT<T>{1.0},
             pybind11::arg("force_zeros") = 1);

    pybind11::class_<PyTensorIterator<T>, std::shared_ptr<PyTensorIterator<T>>>(mod, ("PyTensorIterator" + suffix).c_str())
        .def("__next__", &PyTensorIterator<T>::next, pybind11::return_value_policy::reference)
        .def("reversed", &PyTensorIterator<T>::reversed)
        .def("__iter__", [](PyTensorIterator<T> const &copy) { return copy; })
        .def("__reversed__", [](PyTensorIterator<T> const &copy) { return PyTensorIterator<T>(copy, true); });

    return detail::ExportTensorClass<T>(mod, ("RuntimeTensor" + suffix).c_str(), pybind11::buffer_protocol());
}

void export_tensorf(pybind11::module &mod) {
    auto tensor = export_tensor<float>(mod);
    export_tensor_initf(tensor);
    export_tensor_opsf(tensor);
    export_tensor_manipf(tensor);
    export_tensor_queryf(tensor);
}

void export_tensord(pybind11::module &mod) {
    auto tensor = export_tensor<double>(mod);
    export_tensor_initd(tensor);
    export_tensor_opsd(tensor);
    export_tensor_manipd(tensor);
    export_tensor_queryd(tensor);
}

void export_tensorc(pybind11::module &mod) {
    auto tensor = export_tensor<std::complex<float>>(mod);
    export_tensor_initc(tensor);
    export_tensor_opsc(tensor);
    export_tensor_manipc(tensor);
    export_tensor_queryc(tensor);
}

void export_tensorz(pybind11::module &mod) {
    auto tensor = export_tensor<std::complex<double>>(mod);
    export_tensor_initz(tensor);
    export_tensor_opsz(tensor);
    export_tensor_manipz(tensor);
    export_tensor_queryz(tensor);
}

} // namespace einsums::python