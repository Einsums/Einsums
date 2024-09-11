#include "einsums/Python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::python)
END_EINSUMS_NAMESPACE_CPP(einsums::python)

namespace py = pybind11;
using namespace einsums::python;

void einsums::python::export_python_base(pybind11::module_ &mod) {
    py::enum_<detail::PyPlanUnit>(mod, "PlanUnit")
        .value("CPU", detail::PyPlanUnit::CPU)
        .value("GPU_MAP", detail::PyPlanUnit::GPU_MAP)
        .value("GPU_COPY", detail::PyPlanUnit::GPU_COPY)
        .export_values();
}

PYBIND11_MODULE(einsums_py, mod) {
    mod.doc() = "Einsums Python plugin. Provides a way to interact with the Einsums library through Python.";

    export_python_base(mod);
    export_tensor_algebra(mod);
}