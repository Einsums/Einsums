#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Runtime.hpp>

#include <EinsumsPy/Core/Export.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

bool gpu_enabled() {
#ifdef EINSUMS_COMPUTE_CODE
    return true;
#else
    return false;
#endif
}

void export_Core(py::module_ &mod) {
    mod.def("gpu_enabled", gpu_enabled)
        .def("initialize", einsums::initialize)
        .def("finalize", [](std::string file_name) { einsums::finalize(file_name); }, py::arg("file_name"))
        .def("finalize", [](bool timer_report) { einsums::finalize(timer_report); }, py::arg("timer_report") = false);
}