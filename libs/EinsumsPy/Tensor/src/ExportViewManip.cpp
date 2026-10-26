//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_view_manip(detail::ExportTensorViewClass<T> &tensor) {
    tensor.def("lock", &RuntimeTensorView<T>::lock)
        .def("unlock", &RuntimeTensorView<T>::unlock)
        .def("try_lock", &RuntimeTensorView<T>::try_lock)
        .def("tensor_to_gpu", &RuntimeTensorView<T>::tensor_to_gpu)
        .def("tensor_from_gpu", &RuntimeTensorView<T>::tensor_from_gpu)
        .def("gpu_cache_tensor", [](PyTensorView<T> &self) { return self.gpu_cache_tensor(); })
        .def("gpu_cache_tensor_nowrite", [](PyTensorView<T> &self) { return self.gpu_cache_tensor_nowrite(); })
        .def("gpu_is_expired", &RuntimeTensorView<T>::gpu_is_expired);
}

void export_tensor_view_manipf(detail::ExportTensorViewClass<float> &tensor) {
    export_tensor_view_manip<float>(tensor);
}

void export_tensor_view_manipd(detail::ExportTensorViewClass<double> &tensor) {
    export_tensor_view_manip<double>(tensor);
}

void export_tensor_view_manipc(detail::ExportTensorViewClass<std::complex<float>> &tensor) {
    export_tensor_view_manip<std::complex<float>>(tensor);
}

void export_tensor_view_manipz(detail::ExportTensorViewClass<std::complex<double>> &tensor) {
    export_tensor_view_manip<std::complex<double>>(tensor);
}

} // namespace einsums::python