//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_manip(detail::ExportTensorClass<T> &tensor) {
    tensor.def("lock", &RuntimeTensor<T>::lock)
        .def("unlock", &RuntimeTensor<T>::unlock)
        .def("try_lock", &RuntimeTensor<T>::try_lock)
        .def("tensor_to_gpu", &RuntimeTensor<T>::tensor_to_gpu)
        .def("tensor_from_gpu", &RuntimeTensor<T>::tensor_from_gpu)
        .def("gpu_cache_tensor", [](PyTensor<T> &self) { return self.gpu_cache_tensor(); })
        .def("gpu_cache_tensor_nowrite", [](PyTensor<T> &self) { return self.gpu_cache_tensor_nowrite(); })
        .def("gpu_is_expired", &RuntimeTensor<T>::gpu_is_expired);
}

void export_tensor_manipf(detail::ExportTensorClass<float> &tensor) {
    export_tensor_manip<float>(tensor);
}

void export_tensor_manipd(detail::ExportTensorClass<double> &tensor) {
    export_tensor_manip<double>(tensor);
}

void export_tensor_manipc(detail::ExportTensorClass<std::complex<float>> &tensor) {
    export_tensor_manip<std::complex<float>>(tensor);
}

void export_tensor_manipz(detail::ExportTensorClass<std::complex<double>> &tensor) {
    export_tensor_manip<std::complex<double>>(tensor);
}

} // namespace einsums::python