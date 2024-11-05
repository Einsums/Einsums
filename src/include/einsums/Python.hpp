#pragma once

#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include "einsums/_Common.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::python)

EINSUMS_EXPORT void export_python_base(pybind11::module_ &mod);

#ifdef __HIP__
EINSUMS_EXPORT void export_gpu_except(pybind11::module_ &mod);
#    ifdef EINSUMS_ENABLE_TESTING
EINSUMS_EXPORT void export_python_testing_gpu(pybind11::module_ &mod);
#    endif
#endif

#ifdef EINSUMS_ENABLE_TESTING
EINSUMS_EXPORT void export_python_testing(pybind11::module_ &mod);
#endif

END_EINSUMS_NAMESPACE_HPP(einsums::python)

#include "einsums/python/PyTensor.hpp"
#include "einsums/python/PyTensorAlgebra.hpp"

#ifdef __HIP__
#    include "einsums/python/PyGPUView.hpp"
#endif
