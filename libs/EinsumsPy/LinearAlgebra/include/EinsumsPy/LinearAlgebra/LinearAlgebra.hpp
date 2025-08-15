//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace einsums {
namespace python {
namespace detail {

pybind11::object EINSUMS_EXPORT sum_square(pybind11::buffer const &A);

void EINSUMS_EXPORT gemm(std::string const &transA, std::string const &transB, pybind11::object const &alpha, pybind11::buffer const &A,
                         pybind11::buffer const &B, pybind11::object const &beta, pybind11::buffer &C);

void EINSUMS_EXPORT gemv(std::string const &transA, pybind11::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B,
                         pybind11::object const &beta, pybind11::buffer &C);

void EINSUMS_EXPORT syev(std::string const &jobz, pybind11::buffer &A, pybind11::buffer &W);

void EINSUMS_EXPORT geev(pybind11::buffer &A, pybind11::buffer &W, std::variant<pybind11::buffer, pybind11::none> &Vl,
                         std::variant<pybind11::buffer, pybind11::none> &Vr);

void EINSUMS_EXPORT gesv(pybind11::buffer &A, pybind11::buffer &B);

void EINSUMS_EXPORT scale(pybind11::object factor, pybind11::buffer &A);

void EINSUMS_EXPORT scale_row(int row, pybind11::object factor, pybind11::buffer &A);

void EINSUMS_EXPORT scale_column(int column, pybind11::object factor, pybind11::buffer &A);

pybind11::object EINSUMS_EXPORT dot(pybind11::buffer const &A, pybind11::buffer const &B);

pybind11::object EINSUMS_EXPORT true_dot(pybind11::buffer const &A, pybind11::buffer const &B);

void EINSUMS_EXPORT axpy(pybind11::object const &alpha, pybind11::buffer const &x, pybind11::buffer &y);

void EINSUMS_EXPORT axpby(pybind11::object const &alpha, pybind11::buffer const &x, pybind11::object const &beta, pybind11::buffer &y);

void EINSUMS_EXPORT ger(pybind11::object const &alpha, pybind11::buffer const &X, pybind11::buffer const &Y, pybind11::buffer &A);

std::vector<blas::int_t> EINSUMS_EXPORT getrf(pybind11::buffer &A);

pybind11::tuple EINSUMS_EXPORT extract_plu(pybind11::buffer const &A, std::vector<blas::int_t> const &pivot);

void EINSUMS_EXPORT getri(pybind11::buffer &A, std::vector<blas::int_t> &pivot);

void EINSUMS_EXPORT invert(pybind11::buffer &A);

pybind11::object EINSUMS_EXPORT norm(einsums::linear_algebra::Norm type, pybind11::buffer const &A);

pybind11::object EINSUMS_EXPORT vec_norm(pybind11::buffer const &A);

pybind11::tuple EINSUMS_EXPORT svd(pybind11::buffer const &A);

pybind11::object EINSUMS_EXPORT svd_nullspace(pybind11::buffer const &A);

pybind11::tuple EINSUMS_EXPORT svd_dd(pybind11::buffer const &A, einsums::linear_algebra::Vectors job);

pybind11::tuple EINSUMS_EXPORT truncated_svd(pybind11::buffer const &A, size_t k);

pybind11::tuple EINSUMS_EXPORT truncated_syev(pybind11::buffer const &A, size_t k);

pybind11::object EINSUMS_EXPORT pseudoinverse(pybind11::buffer const &A, pybind11::object const &tol);

pybind11::object EINSUMS_EXPORT solve_continuous_lyapunov(pybind11::buffer const &A, pybind11::buffer const &Q);

pybind11::tuple EINSUMS_EXPORT qr(pybind11::buffer const &A);

void EINSUMS_EXPORT direct_product(pybind11::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B,
                                   pybind11::object const &beta, pybind11::buffer &C);

pybind11::object EINSUMS_EXPORT det(pybind11::buffer const &A);

} // namespace detail
} // namespace python
} // namespace einsums