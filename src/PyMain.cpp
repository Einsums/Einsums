#include <pybind11/pybind11.h>
#include <numpy/

#include "einsums.hpp"

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::python)
END_EINSUMS_NAMESPACE_CPP(einsums::python)

namespace py = pybind11;

PYBIND11_MODULE(einsums_py, mod) {
    mod.doc() = "Einsums Python plugin. Provides a way to interact with the Einsums library through Python.";

    
}