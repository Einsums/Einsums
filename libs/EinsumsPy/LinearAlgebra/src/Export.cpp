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
        .value("MAXABS", einsums::linear_algebra::Norm::MaxAbs)
        .value("ONE", einsums::linear_algebra::Norm::One)
        .value("INFINITY", einsums::linear_algebra::Norm::Infinity)
        .value("FROBENIUS", einsums::linear_algebra::Norm::Frobenius)
        .export_values();

    py::enum_<einsums::linear_algebra::Vectors>(mod, "Vectors")
        .value("ALL", einsums::linear_algebra::Vectors::All)
        .value("SOME", einsums::linear_algebra::Vectors::Some)
        .value("OVERWRITE", einsums::linear_algebra::Vectors::Overwrite)
        .value("NONE", einsums::linear_algebra::Vectors::None)
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
        .def("heev", &syev, "Computes the eigenvectors and eigenvalues of a symmetric or hermitian matrix.")
        .def("geev", &geev, "Computes the eigenvalues and eigenvectors of a general matrix.")
        .def("gesv", &gesv, "Solves a set of linear systems.")
        .def("scale", &scale, "Scales a tensor.")
        .def("scale_row", &scale_row, "Scales a row of a matrix.")
        .def("scale_column", &scale_column, "Scales a column of a matrix.")
        .def("dot", &dot,
             "Computes the sum of products of elements. This is the dot product for real inputs, but not the true dot product for complex "
             "inputs, as the true dot product uses the conjugate of the first tensor.")
        .def("true_dot", &true_dot,
             "Computes the true dot product. This means that the complex conjugate of the first tensor will be used. For real inputs, this "
             "is the same as the other dot product function.")
        .def("axpy", &axpy, "Scale a tensor and add it to another.")
        .def("axpby", &axpby, "Scale two tensors and add them together.")
        .def("ger", &ger, "Perform the rank-1 update.")
        .def("getrf", &getrf, "Perform the setup for LU decomposition.")
        .def("extract_plu", &extract_plu, "Extract the permutation, lower triangular, and upper triangular matrices after a call to getrf.")
        .def("getri", &getri, "Find the matrix inverse after a call to getrf.")
        .def("invert", &invert, "Compute the matrix inverse. Internally, this calls getrf then getri.")
        .def("norm", &norm, "Compute the matrix norm.")
        .def("vec_norm", &vec_norm, "Compute the vector norm.")
        .def("svd", &svd, "Perform singular value decomposition.")
        .def("svd_nullspace", &svd_nullspace, "Find the nullspace using singular value decomposition.")
        .def("svd_dd", &svd_dd, py::arg("A"), py::arg("job") = einsums::linear_algebra::Vectors::All,
             "Perform singular value decomposition using the divide and conquer algorithm.")
        .def("truncated_svd", &truncated_svd, "Perform singular value decomposition but ignore some number of small singular values.")
        .def("truncated_syev", &truncated_syev, "Perform symmetric/hermitian eigendecomposition but ignore some number of small eigenvalues.")
        .def("pseudoinverse", &pseudoinverse, "Compute the pseudoinverse of a non-invertible matrix.")
        .def("solve_continuous_lyapunov", &solve_continuous_lyapunov, "Solve a continuous Lyapunov equation.")
        .def("qr", &qr, "Set up for QR decomposition of a matrix.")
        .def("q", &q, "Extract the Q matrix after a call to qr.")
        .def("r", &r, "Extract the R matrix after a call to qr.")
        .def("direct_product", &direct_product, "Compute the direct product between two tensors.")
        .def("det", &det, "Compute the matrix determinant.");
}