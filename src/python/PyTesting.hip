#include "einsums/_GPUUtils.hpp"

#include "einsums/Python.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace py = pybind11;

using namespace einsums;
using namespace einsums::python;

void einsums::python::export_python_testing_gpu(pybind11::module_ &mod) {
    mod.def("throw_hip", [](int status) {
        gpu::hip_catch((hipError_t)status); })
    .def("throw_hipblas", [](int status) {
        gpu::hipblas_catch((hipblasStatus_t)status); })
    .def("throw_hipsolver", [](int status) {
        gpu::hipsolver_catch((hipsolverStatus_t)status);});
}