#include "einsums/_Common.hpp"

#include "einsums/Python.hpp"
#include "einsums/Timer.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::python)
END_EINSUMS_NAMESPACE_CPP(einsums::python)

namespace py = pybind11;
using namespace einsums::python;

static const auto einsums_docstring =
    "einsums_py module\n"
    "-----------------\n\n"
    "Enables interaction with the main ideas of the Einsums library from Python. "
    "Due to limitations with how templates are handled, this module can not directly interact "
    "with the einsums::Tensor<T> family of classes and instead need to have their own. "
    "These Python-compatible tensors are fully compatible with the einsum::Tensor<T> family, "
    "but we can not provide compile-time optimizations in all cases on the C++ side.\n"
    "In order to emulate the compile-time optimization from the Python side, einsums_py provides "
    "a function to compile a contraction into a plan. This compile expression does all that the "
    "compiler would do for the einsums::tensor_algebra::einsum call, but it does it at runtime, and "
    "it does not actually do any contraction itself. The plan it produces can then be used on the tensors "
    "and prefactors to perform the actual contraction.\n"
    "The einsums_py module should be able to use numpy.array objects as input tensors. However, since "
    "numpy only has dense tensors on the CPU, we also provide functionality similar to the Einsums collected "
    "tensors, including BlockTensors and TiledTensors. There is also a way to create a GPUView of a CPU buffer object "
    "so that you can use graphics card acceleration from Python. In order to use this, einsums and einsums_py need to "
    "be compiled with EINSUMS_BUILD_GPU enabled. To check if this is the case, we provide einsums_py.gpu_enabled(). ";

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
    mod.def("gpu_enabled", gpu_enabled)
        .def("initialize", einsums::initialize)
        .def(
            "finalize", [](bool timer_report) { einsums::finalize(timer_report); }, py::arg("timer_report") = false)
        .def(
            "finalize", [](std::string timer_report) { einsums::finalize(timer_report); }, py::arg("output_file"))
        .def("report", []() { einsums::timer::report(); })
        .def("report", [](std::string output_file) { einsums::timer::report(output_file); }, py::arg("output_file"));

    py::register_exception<einsums::EinsumsException>(mod, "CoreEinsumsException");
}

PYBIND11_MODULE(core, mod) {
    einsums::initialize();

    mod.doc() = einsums_docstring;

    export_python_base(mod);
    export_tensor_algebra(mod);
#ifdef __HIP__
    export_gpu_view(mod);
#endif
    export_tensor<float>(mod);
    export_tensor<double>(mod);
    export_tensor<std::complex<float>>(mod);
    export_tensor<std::complex<double>>(mod);

#ifdef EINSUMS_ENABLE_TESTING
    export_python_testing(mod);
#    ifdef __HIP__
    export_python_testing_gpu(mod);
#    endif
#endif

    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() -> void { einsums::finalize(); }));
}