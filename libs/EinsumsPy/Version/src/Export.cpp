//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Version.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

EINSUMS_EXPORT void export_Version(py::module_ &mod) {

    py::module version = mod.def_submodule("version", "C++ Einsums Version sub-module");

    version.def("major", &einsums::major_version, "Returns the major einsums version.");
    version.def("minor", &einsums::minor_version, "Returns the minor einsums version.");
    version.def("patch", &einsums::patch_version, "Returns the patch einsums version.");
    version.def("tag", &einsums::tag, "Returns the version tag.");
    version.def("complete_version", &einsums::complete_version, "Returns the full einsums version.");
    version.def("configuration_string", &einsums::configuration_string, "Returns the full einsums version.");
    version.def("build_string", &einsums::build_string, "Returns the einsums version string.");
    version.def("build_type", &einsums::build_type, "Returns the einsums build type ('Debug', 'Release', etc.)");
    version.def("build_date_time", &einsums::build_date_time, "Returns the einsums build date and time.");
    version.def("full_build_string", &einsums::full_build_string);
    version.def("copyright", &einsums::copyright);
}