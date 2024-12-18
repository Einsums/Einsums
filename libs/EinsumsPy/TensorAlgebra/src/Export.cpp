#include <EinsumsPy/TensorAlgebra/Export.hpp>
#include <EinsumsPy/TensorAlgebra/PyTensorAlgebra.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace einsums;
using namespace einsums::tensor_algebra;

void export_TensorAlgebra(py::module_ &mod) {
    mod.def("compile_plan", &compile_plan);
    py::class_<PyEinsumGenericPlan, std::shared_ptr<PyEinsumGenericPlan>>(mod, "EinsumGenericPlan")
        .def("execute",
             [](PyEinsumGenericPlan &self, pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                pybind11::buffer const &A, pybind11::buffer const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#ifdef EINSUMS_COMPUTE_CODE
        .def("execute",
             [](PyEinsumGenericPlan &self, pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                python::PyGPUView const &A, python::PyGPUView const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#endif
        ;

    py::class_<PyEinsumDotPlan, PyEinsumGenericPlan, std::shared_ptr<PyEinsumDotPlan>>(mod, "EinsumDotPlan")
        .def("execute",
             [](PyEinsumDotPlan &self, pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                pybind11::buffer const &A, pybind11::buffer const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#ifdef EINSUMS_COMPUTE_CODE
        .def("execute",
             [](PyEinsumDotPlan &self, pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                python::PyGPUView const &A, python::PyGPUView const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#endif
        ;

    py::class_<PyEinsumDirectProductPlan, PyEinsumGenericPlan, std::shared_ptr<PyEinsumDirectProductPlan>>(mod, "EinsumDirectProductPlan")
        .def("execute", [](PyEinsumDirectProductPlan &self, pybind11::object const &C_prefactor, pybind11::buffer &C,
                           pybind11::object const &AB_prefactor, pybind11::buffer const &A,
                           pybind11::buffer const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#ifdef EINSUMS_COMPUTE_CODE
        .def("execute", [](PyEinsumDirectProductPlan &self, pybind11::object const &C_prefactor, python::PyGPUView &C,
                           pybind11::object const &AB_prefactor, python::PyGPUView const &A,
                           python::PyGPUView const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#endif
        ;

    py::class_<PyEinsumGerPlan, PyEinsumGenericPlan, std::shared_ptr<PyEinsumGerPlan>>(mod, "EinsumGerPlan")
        .def("execute",
             [](PyEinsumGerPlan &self, pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                pybind11::buffer const &A, pybind11::buffer const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#ifdef EINSUMS_COMPUTE_CODE
        .def("execute",
             [](PyEinsumGerPlan &self, pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                python::PyGPUView const &A, python::PyGPUView const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#endif
        ;

    py::class_<PyEinsumGemvPlan, PyEinsumGenericPlan, std::shared_ptr<PyEinsumGemvPlan>>(mod, "EinsumGemvPlan")
        .def("execute",
             [](PyEinsumGemvPlan &self, pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                pybind11::buffer const &A, pybind11::buffer const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#ifdef EINSUMS_COMPUTE_CODE
        .def("execute",
             [](PyEinsumGemvPlan &self, pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                python::PyGPUView const &A, python::PyGPUView const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#endif
        ;

    py::class_<PyEinsumGemmPlan, PyEinsumGenericPlan, std::shared_ptr<PyEinsumGemmPlan>>(mod, "EinsumGemmPlan")
        .def("execute",
             [](PyEinsumGemmPlan &self, pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                pybind11::buffer const &A, pybind11::buffer const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#ifdef EINSUMS_COMPUTE_CODE
        .def("execute",
             [](PyEinsumGemmPlan &self, pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                python::PyGPUView const &A, python::PyGPUView const &B) { self.execute(C_prefactor, C, AB_prefactor, A, B); })
#endif
        ;
}