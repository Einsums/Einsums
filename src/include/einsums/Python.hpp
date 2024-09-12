#pragma once

#include "einsums/_Common.hpp"
#include <pybind11/pybind11.h>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::python)

namespace detail {

enum PyPlanUnit { CPU, GPU_MAP, GPU_COPY };

} // namespace detail

EINSUMS_EXPORT void export_python_base(pybind11::module_ &mod);

#ifdef __HIP__
EINSUMS_EXPORT void export_gpu(pybind11::module_ &mod);
#endif

END_EINSUMS_NAMESPACE_HPP(einsums::python)

#include "einsums/python/PyTensorAlgebra.hpp"