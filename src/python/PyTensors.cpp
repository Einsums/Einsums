#include "einsums/RuntimeTensor.hpp"
#include "einsums/python/PyTensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "einsums.hpp"

namespace py = pybind11;

using namespace einsums;
using namespace einsums::python;

void einsums::python::export_tensor_typeless(pybind11::module_ &mod) {
    pybind11::class_<einsums::detail::RuntimeTensorNoType, std::shared_ptr<einsums::detail::RuntimeTensorNoType>>(mod, "RuntimeTensor");
    pybind11::class_<einsums::detail::RuntimeTensorViewNoType, std::shared_ptr<einsums::detail::RuntimeTensorViewNoType>>(
        mod, "RuntimeTensorView");
}
