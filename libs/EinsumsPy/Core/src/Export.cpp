//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Types.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Runtime.hpp>

#include <EinsumsPy/Core/Export.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace einsums;

bool gpu_enabled() {
#ifdef EINSUMS_COMPUTE_CODE
    return true;
#else
    return false;
#endif
}

void export_Core(py::module_ &mod) {
    mod.def("gpu_enabled", gpu_enabled)
        // .def("initialize", einsums::initialize)
        // .def(
        // "finalize", [](std::string file_name) { einsums::finalize(file_name); }, py::arg("file_name"))
        // .def("finalize", [](bool timer_report) { einsums::finalize(timer_report); }, py::arg("timer_report") = false)
        ;

    auto config_map = py::class_<einsums::GlobalConfigMap, std::shared_ptr<einsums::GlobalConfigMap>>(mod, "GlobalConfigMap");

    config_map.def_static("get_singleton", einsums::GlobalConfigMap::get_singleton)
        .def("empty", [](GlobalConfigMap &self) { return self.empty(); })
        .def("size", [](GlobalConfigMap &self) { return self.size(); })
        .def("max_size", [](GlobalConfigMap &self) { return self.max_size(); })
        .def("clear", [](GlobalConfigMap &self) { self.clear(); })
        .def("erase", [](GlobalConfigMap &self, std::string const &str) { return self.erase(str); })
        .def("at_string", [](GlobalConfigMap &self, std::string const &str) { return self.at_string(str); })
        .def("at_int", [](GlobalConfigMap &self, std::string const &str) { return self.at_int(str); })
        .def("at_double", [](GlobalConfigMap &self, std::string const &str) { return self.at_double(str); })
        .def("get_string", [](GlobalConfigMap &self, std::string const &str) { return self.get_string(str); })
        .def("get_int", [](GlobalConfigMap &self, std::string const &str) { return self.get_int(str); })
        .def("get_double", [](GlobalConfigMap &self, std::string const &str) { return self.get_double(str); })
        .def("set_string", [](GlobalConfigMap &self, std::string const &str, std::string &val) { self.get_string(str) = val; })
        .def("set_int", [](GlobalConfigMap &self, std::string const &str, std::int64_t val) { self.get_int(str) = val; })
        .def("set_double", [](GlobalConfigMap &self, std::string const &str, double val) { self.get_double(str) = val; });
}