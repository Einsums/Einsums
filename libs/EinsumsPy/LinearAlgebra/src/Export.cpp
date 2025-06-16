//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Utilities/InCollection.hpp>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

} // namespace detail
} // namespace python
} // namespace einsums

EINSUMS_EXPORT void export_LinearAlgebra(py::module_ &mod) {

}