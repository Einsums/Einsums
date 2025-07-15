//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/Diagonal.hpp>

#include <numbers>
#include <string>

namespace einsums {

/**
 * Create a random positive or negative definite matrix.
 * A positive definite matrix is a symmetric matrix whose eigenvalues are all positive.
 * Similarly for negative definite matrices.
 *
 * This function first generates a set of random eigenvectors, making sure they are non-singular.
 * Then, it uses these to form an orthonormal eigenbasis for the new matrix. Then, it generates
 * the eigenvalues. The eigenvalues are distributed using a Maxwell-Boltzmann distribution
 * with the given mean, defaulting to 1. Then, the returned matrix is formed
 * by computing @f$P^TDP@f$. If the mean is negative, then the result will be a negative
 * definite matrix.
 *
 * @param name The name for the matrix.
 * @param rows The number of rows.
 * @param cols The number of columns. Should equal the number of rows.
 * @param mean The mean for the eigenvalues. Defaults to 1.
 * @return A new positive definite or negative definite matrix.
 */
template <typename T = double, typename Distribution>
    requires requires(Distribution dist) {
        { dist(einsums::random_engine) } -> std::same_as<T>;
    }
auto create_random_unitary(std::string const &name, Distribution &&dist, int rows, int cols) -> Tensor<T, 2> {
    if (rows != cols) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only make square positive definite matrices.");
    }
    Tensor<T, 2> unitary(name, rows, cols);

    Tensor<T, 2>             Temp = unitary;
    std::vector<blas::int_t> pivs;

    // Make sure the eigenvectors are non-singular.
    do {
        unitary = create_random_tensor<T>("name", std::forward<Distribution>(dist), rows, cols);
        Temp    = unitary;
    } while (linear_algebra::getrf(&Temp, &pivs) > 0);

    // QR decompose Evecs to get a random matrix of orthonormal eigenvectors.
    auto pair = linear_algebra::qr(unitary);

    return linear_algebra::q(std::get<0>(pair), std::get<1>(pair));
}

template <typename T = double>
auto create_random_unitary(std::string const &name, int rows, int cols) -> Tensor<T, 2> {
    if constexpr (IsComplexV<T>) {
        return create_random_unitary<T>(name, detail::unit_circle_distribution<T>(), rows, cols);
    } else {
        return create_random_unitary<T>(name, std::uniform_real_distribution<T>(-1, 1), rows, cols);
    }
}

} // namespace einsums