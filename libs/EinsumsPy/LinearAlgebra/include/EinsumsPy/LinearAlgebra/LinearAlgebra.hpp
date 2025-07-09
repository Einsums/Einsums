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

pybind11::tuple EINSUMS_EXPORT sum_square(pybind11::buffer const &A);

void EINSUMS_EXPORT gemm(std::string const &transA, std::string const &transB, pybind11::object const &alpha, pybind11::buffer const &A,
                         pybind11::buffer const &B, pybind11::object const &beta, pybind11::buffer &C);

void EINSUMS_EXPORT gemv(std::string const &transA, pybind11::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B,
                         pybind11::object const &beta, pybind11::buffer &C);

void EINSUMS_EXPORT syev(std::string const &jobz, pybind11::buffer &A, pybind11::buffer &W);

void EINSUMS_EXPORT geev(std::string const &jobvl, std::string const &jobvr, pybind11::buffer &A, pybind11::buffer &W,
                         std::variant<pybind11::buffer, pybind11::none> &Vl, std::variant<pybind11::buffer, pybind11::none> &Vr);

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

pybind11::object EINSUMS_EXPORT q(pybind11::buffer const &qr, pybind11::buffer const &tau);

pybind11::object EINSUMS_EXPORT r(pybind11::buffer const &qr, pybind11::buffer const &tau);

void EINSUMS_EXPORT direct_product(pybind11::object const &alpha, pybind11::buffer const &A, pybind11::buffer const &B,
                                   pybind11::object const &beta, pybind11::buffer &C);

pybind11::object EINSUMS_EXPORT det(pybind11::buffer const &A);

/**
 * @brief Determines how to loop over vector calls for tensors that don't have contiguous strides.
 *
 * How this information is used is to perform a vectorized loop up to @c hard_elems and turn this into an offset for the buffer pointer
 * using the @c hard_index_strides and @c X_strides . This will detect if the innermost stride is non-unitary and adjust as needed. From
 * there, simply pass @c easy_elems as the size of the vector to a BLAS call and @c incx as the increment. The index strides will contain
 * all of the index strides for the whole tensor, so it will need to be recalculated. Use the return value from this function to pass into
 * @c recalc_index_strides to perform this calculation.
 *
 * @param X The buffer to parse.
 * @param[out] easy_elems The number of elements that can be handled by a BLAS call directly.
 * @param[out] incx The step size for the BLAS call.
 * @param[out] hard_elems The number of elemenst for the outer loop. If it is zero, then the whole tensor can be handled directly by the
 * BLAS call.
 * @param[out] X_strides The strides for the buffer. Because the input buffer's strides are in bytes, they are rescaled into elements and
 * put into this vector.
 * @param[out] hard_index_strides The strides for the hard loop. These will need to be recalculated before use.
 * @return The number of elements to keep in the index strides.
 */
int EINSUMS_EXPORT determine_easy_vector(pybind11::buffer const &X, size_t *easy_elems, size_t *incx, size_t *hard_elems,
                                         std::vector<size_t> *X_strides, std::vector<size_t> *hard_index_strides);

/**
 * @brief Shrinks a stride vector and recalculates so that the last element is 1.
 *
 * This will go through and divide each element by the last element.
 *
 * @param[in,out] index_strides The stride vector to recalculate.
 * @param[in] new_rank The new size of the vector.
 */
void EINSUMS_EXPORT recalc_index_strides(std::vector<size_t> *index_strides, int new_rank);

} // namespace detail
} // namespace python
} // namespace einsums