#pragma once

#include <Einsums/Config.hpp>
#include <pybind11/pybind11.h>

EINSUMS_EXPORT void export_GPU(pybind11::module_ &mod);