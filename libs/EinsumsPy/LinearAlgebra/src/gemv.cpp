//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Utilities/InCollection.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "EinsumsPy/Tensor/PyTensor.hpp"
#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void gemv(std::string const &transA, py::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B, py::object const &beta,
          pybind11::buffer &C) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false), C_info = C.request(true);

    if (transA.length() < 1) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "The transpose string needs to have data! It can not be empty!");
    }

    if (A_info.format != B_info.format || A_info.format != C_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only perform gemv on tensors with the same stored type! Got A ({}), B ({}), C ({}).",
                                A_info.format, B_info.format, C_info.format);
    }

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(),
                           [&]() {
                               auto A_tens = buffer_to_tensor<Float>(A);
                               auto B_tens = buffer_to_tensor<Float>(B);
                               auto C_tens = buffer_to_tensor<Float>(C);
                               einsums::linear_algebra::detail::gemv(transA[0], alpha.cast<Float>(), A_tens, B_tens, beta.cast<Float>(),
                                                                     &C_tens);
                           }())
    else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only perform gemv on floating point matrices! Got type {}.", A_info.format);
    }
}

} // namespace detail
} // namespace python
} // namespace einsums