#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include "einsums/_GPUUtils.hpp"

#include "einsums/Python.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace py = pybind11;

using namespace einsums;
using namespace einsums::python;

void einsums::python::export_python_testing_gpu(pybind11::module_ &mod) {
    mod.def("throw_hip", [](int status, bool throw_success = false) {
        gpu::hip_catch((hipError_t)status, throw_success); })
    .def("throw_hipblas", [](int status, bool throw_success = false) {
        gpu::hipblas_catch((hipblasStatus_t)status, throw_success); })
    .def("throw_hipsolver", [](int status, bool throw_success = false) {
        gpu::hipsolver_catch((hipsolverStatus_t)status, throw_success);});
}