//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.hpp"

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

py::object sum_square(pybind11::buffer const &A) {
    pybind11::buffer_info A_info = A.request();

    py::object out;

    EINSUMS_PY_LINALG_CALL(A_info.item_type_is_equivalent_to<Float>(), [&]() {
        RemoveComplexT<Float> sumsq{0.0};
        RemoveComplexT<Float> scale{0.0};
        einsums::linear_algebra::detail::sum_square(buffer_to_tensor<Float>(A), &scale, &sumsq);
        out = py::cast(sumsq * scale * scale);
    }())
    else {
        EINSUMS_THROW_EXCEPTION(pybind11::value_error, "Could not perform sum_square on the requested type {}!", A_info.format);
    }

    return out;
}

} // namespace detail
} // namespace python
} // namespace einsums