//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Types.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/Section.hpp>
#include <Einsums/Runtime.hpp>

#include <exception>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>

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
    mod.def("gpu_enabled", gpu_enabled, "Check if Einsums was compiled with GPU capabilities.")
        .def(
            "initialize", [](std::vector<std::string> &argv) { einsums::initialize(argv); }, "Initialize the Einsums module.")
        .def("finalize", einsums::finalize, "Clean up the Einsums module.")
        .def(
            "log", [](int level, std::string const &str) { EINSUMS_LOG(level, str); }, "Log a message at the given log level.")
        .def(
            "log_trace", [](std::string const &str) { EINSUMS_LOG_TRACE(str); }, "Log a message at the trace level.")
        .def(
            "log_debug", [](std::string const &str) { EINSUMS_LOG_DEBUG(str); }, "Log a message at the debug level.")
        .def(
            "log_info", [](std::string const &str) { EINSUMS_LOG_INFO(str); }, "Log a message at the info level.")
        .def(
            "log_warn", [](std::string const &str) { EINSUMS_LOG_WARN(str); }, "Log a message at the warning level.")
        .def(
            "log_error", [](std::string const &str) { EINSUMS_LOG_ERROR(str); }, "Log a message at the error level.")
        .def(
            "log_critical", [](std::string const &str) { EINSUMS_LOG_CRITICAL(str); }, "Log a message at the critical level.");

    auto config_map = py::class_<einsums::GlobalConfigMap, std::shared_ptr<einsums::GlobalConfigMap>>(
        mod, "GlobalConfigMap", "Contains all of the options handled by Einsums.");

    config_map.def_static("get_singleton", einsums::GlobalConfigMap::get_singleton, "Get the single unique instance.")
        .def(
            "empty", [](GlobalConfigMap &self) { return self.empty(); }, "Check to see if the map is empty.")
        .def(
            "size", [](GlobalConfigMap &self) { return self.size(); }, "Get the number of elements in the map.")
        .def(
            "max_size", [](GlobalConfigMap &self) { return self.max_size(); }, "Get the maximum size of the map.")
        .def(
            "get_str", [](GlobalConfigMap &self, std::string const &str) { return self.get_string(str); }, "Get a string option.")
        .def(
            "get_int", [](GlobalConfigMap &self, std::string const &str) { return self.get_int(str); }, "Get an integer option.")
        .def(
            "get_float", [](GlobalConfigMap &self, std::string const &str) { return self.get_double(str); }, "Get a floating point option.")
        .def(
            "get_bool", [](GlobalConfigMap &self, std::string const &str) { return self.get_bool(str); }, "Get a Boolean option.")
        .def(
            "set_str",
            [](GlobalConfigMap &self, std::string const &key, std::string const &value) {
                auto guard                              = std::lock_guard(self);
                self.get_string_map()->get_value()[key] = value;
            },
            "Set a string option.")
        .def(
            "set_int",
            [](GlobalConfigMap &self, std::string const &key, std::int64_t value) {
                auto guard                           = std::lock_guard(self);
                self.get_int_map()->get_value()[key] = value;
            },
            "Set an integer option.")
        .def(
            "set_float",
            [](GlobalConfigMap &self, std::string const &key, double value) {
                auto guard                              = std::lock_guard(self);
                self.get_double_map()->get_value()[key] = value;
            },
            "Set a floating point option.")
        .def(
            "set_bool",
            [](GlobalConfigMap &self, std::string const &key, bool value) {
                auto guard                            = std::lock_guard(self);
                self.get_bool_map()->get_value()[key] = value;
            },
            "Set a Boolean option.");

    auto section = py::class_<einsums::Section>(mod, "Section", "Represents a section in the profiling report.");

    section.def(py::init<std::string const &>())
        .def(py::init<std::string const &, bool>())
        .def(py::init<std::string const &, std::string const &>())
        .def(py::init<std::string const &, std::string const &, bool>())
        .def("end", &Section::end, "End a section early. Usually ended by the destructor.");
}