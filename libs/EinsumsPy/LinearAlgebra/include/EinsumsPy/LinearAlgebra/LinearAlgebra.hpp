//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace einsums {
namespace python {
namespace detail {

void EINSUMS_EXPORT sum_square(pybind11::buffer const &A, pybind11::object &scale, pybind11::object &sum_sq);

void EINSUMS_EXPORT gemm(std::string const &transA, std::string const &transB, pybind11::object const &alpha, pybind11::buffer const &A,
                         pybind11::buffer const &B, pybind11::object const &beta, pybind11::buffer &C);

void EINSUMS_EXPORT gemv(std::string const &transA, pybind11::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B,
                         pybind11::object const &beta, pybind11::buffer &C);

void EINSUMS_EXPORT syev(std::string const &jobz, pybind11::buffer &A, pybind11::buffer &W);

void EINSUMS_EXPORT geev(std::string const &jobvl, std::string const &jobvr, pybind11::buffer &A, pybind11::buffer &W, pybind11::buffer &Vl,
                         pybind11::buffer &Vr);

void EINSUMS_EXPORT gesv(pybind11::buffer &A, pybind11::buffer &B);

void EINSUMS_EXPORT scale(pybind11::object factor, pybind11::buffer &A);

void EINSUMS_EXPORT scale_row(int row, pybind11::object factor, pybind11::buffer &A);

void EINSUMS_EXPORT scale_column(int column, pybind11::object factor, pybind11::buffer &A);

pybind11::object EINSUMS_EXPORT dot(pybind11::buffer const &A, pybind11::buffer const &B);

pybind11::object EINSUMS_EXPORT true_dot(pybind11::buffer const &A, pybind11::buffer const &B);

} // namespace detail
} // namespace python
} // namespace einsums