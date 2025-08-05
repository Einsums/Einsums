//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/Diagonal.hpp>

#include <numbers>
#include <string>

namespace einsums {

/**
 * Create a random positive or negative semi-definite matrix.
 * A positive semi-definite matrix is a symmetric matrix whose eigenvalues are all non-negative.
 * Similarly for negative semi-definite matrices.
 *
 * This function first generates a set of random eigenvectors, making sure they are non-singular.
 * Then, it uses these to form an orthonormal eigenbasis for the new matrix. Then, it generates
 * the eigenvalues. The eigenvalues are distributed using a Maxwell-Boltzmann distribution
 * with the given mean, defaulting to 1. If desired, a number of eigenvalues can be forced to be zero.
 * Then, the returned matrix is formed by computing @f$P^TDP@f$.
 *
 * @param name The name for the matrix.
 * @param rows The number of rows.
 * @param cols The number of columns. Should equal the number of rows.
 * @param mean The mean for the eigenvalues. Defaults to 1. If negative, the result is a negative semi-definite matrix.
 * @param force_zeros The number of elements to force to be zero. Defaults to 1.
 * @return A new positive or negative semi-definite matrix.
 */
template <typename T = double, bool Normalize = false>
auto create_random_semidefinite(std::string const &name, int rows, int cols, RemoveComplexT<T> mean = RemoveComplexT<T>{1.0},
                                int force_zeros = 1) -> Tensor<T, 2> {
    using Real = RemoveComplexT<T>;
    if (rows != cols) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only make square positive definite matrices.");
    }
    Tensor<T, 2> Evecs("name", rows, cols);

    Tensor<T, 2>             Temp = Evecs;
    BufferVector<blas::int_t> pivs;

    // Make sure the eigenvectors are non-singular.
    do {
        Evecs = create_random_tensor<T>("name", rows, cols);
        Temp  = Evecs;
    } while (linear_algebra::getrf(&Temp, &pivs) > 0);

    // QR decompose Evecs to get a random matrix of orthonormal eigenvectors.
    auto pair = linear_algebra::qr(Evecs);

    Evecs = linear_algebra::q(std::get<0>(pair), std::get<1>(pair));

    std::default_random_engine engine;

    // Create random eigenvalues. Need to calculate the standard deviation from the mean.
    auto normal =
        std::normal_distribution<Real>(0, std::abs(mean) / Real{2.0} / std::numbers::sqrt2_v<Real> / std::numbers::inv_sqrtpi_v<Real>);

    Tensor<T, 1> Evals("name2", rows);

    for (int i = 0; i < rows; i++) {
        if (i < force_zeros) {
            Evals(i) = T{0.0};
        } else {
            // Maxwell-Boltzmann distribute the eigenvalues. Make sure they are positive.
            T val1 = normal(engine), val2 = normal(engine), val3 = normal(engine);

            Evals(i) = std::sqrt(val1 * val1 + val2 * val2 + val3 * val3);
            if (mean < Real{0.0}) {
                Evals(i) = -Evals(i);
            }
        }
    }

    // Create the tensor.
    Tensor<T, 2> ret = diagonal(Evals);

    linear_algebra::gemm<false, false>(1.0, ret, Evecs, 0.0, &Temp);
    linear_algebra::gemm<true, false>(1.0, Evecs, Temp, 0.0, &ret);

    ret.set_name(name);

    return ret;
}

} // namespace einsums