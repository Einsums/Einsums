#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "einsums/Python.hpp"
#include "einsums/_Common.hpp"

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::python)
END_EINSUMS_NAMESPACE_CPP(einsums::python)

namespace py = pybind11;
using namespace einsums::python;

#ifdef __HIP__
static bool gpu_enabled() {
    return true;
}
#else
static bool gpu_enabled() {
    return false;
}
#endif

void einsums::python::export_python_base(pybind11::module_ &mod) {
    py::enum_<detail::PyPlanUnit>(mod, "PlanUnit")
        .value("CPU", detail::PyPlanUnit::CPU)
        .value("GPU_MAP", detail::PyPlanUnit::GPU_MAP)
        .value("GPU_COPY", detail::PyPlanUnit::GPU_COPY)
        .export_values();
    mod.def("gpu_enabled", gpu_enabled).def("initialize", einsums::initialize).def("finalize", einsums::finalize, py::arg("timer_report") = false);

    py::register_exception<einsums::EinsumsException>(mod, "CoreEinsumsException");
}

PYBIND11_MODULE(einsums_py, mod) {
    einsums::initialize();

    mod.doc() = "Einsums Python plugin. Provides a way to interact with the Einsums library through Python.";

    export_python_base(mod);
    export_tensor_algebra(mod);
        #ifdef __HIP__
    export_gpu(mod);
    #endif

    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() -> void { einsums::finalize(); }));
}