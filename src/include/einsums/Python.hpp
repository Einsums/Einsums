#pragma once

#include "einsums/_Common.hpp"

#include <pybind11/pybind11.h>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::python)

EINSUMS_EXPORT void export_python_base(pybind11::module_ &mod);

#ifdef __HIP__
EINSUMS_EXPORT void export_gpu(pybind11::module_ &mod);
#endif

END_EINSUMS_NAMESPACE_HPP(einsums::python)

#ifdef __HIP__
#    include "einsums/python/PyGPUView.hpp"
#endif

#include "einsums/python/PyTensor.hpp"
#include "einsums/python/PyTensorAlgebra.hpp"