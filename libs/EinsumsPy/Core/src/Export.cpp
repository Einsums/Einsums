#include <Einsums/Config.hpp>

#include <EinsumsPy/Core/Export.hpp>
#include <pybind11/pybind11.h>
#include <Einsums/Errors/Error.hpp>

namespace py = pybind11;

bool gpu_enabled() {
#ifdef EINSUMS_COMPUTE_CODE
    return true;
#else
    return false;
#endif
}

void export_Core(py::module_ &mod) {
    mod.def("gpu_enabled", gpu_enabled);
}