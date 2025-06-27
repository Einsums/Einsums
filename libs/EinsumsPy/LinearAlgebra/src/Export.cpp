//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Utilities/InCollection.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {} // namespace detail
} // namespace python
} // namespace einsums

using namespace einsums::python::detail;

EINSUMS_EXPORT void export_LinearAlgebra(py::module_ &mod) {
    py::enum_<einsums::linear_algebra::Norm>(mod, "Norm")
        .value("MaxAbs", einsums::linear_algebra::Norm::MaxAbs)
        .value("One", einsums::linear_algebra::Norm::One)
        .value("Infinity", einsums::linear_algebra::Norm::Infinity)
        .value("Frobenius", einsums::linear_algebra::Norm::Frobenius)
        .export_values();

    py::enum_<einsums::linear_algebra::Vectors>(mod, "Vectors")
        .value("All", einsums::linear_algebra::Vectors::All)
        .value("Some", einsums::linear_algebra::Vectors::Some)
        .value("Overwrite", einsums::linear_algebra::Vectors::Overwrite)
        .value("None", einsums::linear_algebra::Vectors::None)
        .export_values();

    mod.def("sum_square", &sum_square,
            "Calculate the sum of the squares of the elements of a vector. The result will be scale**2 * sum_sq. sum_sq is the first "
            "return value, scale is the second.")
        .def("gemm", &gemm,
             "Matrix multiplication. The first two arguments indicate whether to transpose the input matrices. The third is a scale factor "
             "for the input matrices. The next two arguments are the matrices to multiply. The fifth argument is a scale factor for the "
             "output matrix, follwed by the output matrix.")
        .def("gemv", &gemv,
             "Matrix vector multiplication. The first argument is whether to transpose the matrix. Then, the scale factor for the inputs. "
             "Then, the matrix, followed by the input vector. Then the scale factor for the output. Finally, the output vector.")
        .def("syev", &syev, "Computes the eigenvectors and eigenvalues of a symmetric or hermitian matrix.")
        .def("heev", &syev)
        .def("geev", &geev)
        .def("gesv", &gesv)
        .def("scale", &scale)
        .def("scale_row", &scale_row)
        .def("scale_column", &scale_column)
        .def("dot", &dot)
        .def("true_dot", &true_dot)
        .def("axpy", &axpy)
        .def("axpby", &axpby)
        .def("ger", &ger)
        .def("getrf", &getrf)
        .def("getri", &getri)
        .def("invert", &invert)
        .def("norm", &norm)
        .def("vec_norm", &vec_norm)
        .def("svd", &svd)
        .def("svd_nullspace", &svd_nullspace)
        .def("svd_dd", &svd_dd, py::arg("A"), py::arg("job") = einsums::linear_algebra::Vectors::All)
        .def("truncated_svd", &truncated_svd)
        .def("truncated_syev", &truncated_syev)
        .def("pseudoinverse", &pseudoinverse)
        .def("solve_continuous_lyapunov", &solve_continuous_lyapunov)
        .def("qr", &qr)
        .def("q", &q)
        .def("direct_product", &direct_product)
        .def("det", &det);
}