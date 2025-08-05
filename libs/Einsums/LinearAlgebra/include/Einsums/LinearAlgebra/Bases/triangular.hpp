//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums::linear_algebra::detail {

template <typename T, Container Pivots>
int impl_lu_decomp(einsums::detail::TensorImpl<T> &A, Pivots &pivot) {
    size_t const m = A.dim(0), n = A.dim(1), min_dim = std::min(m, n);
    int          ret = 0;

    // Gaussian elimination.
    for (size_t k = 0; k < min_dim - 1; k++) {
        // Find the largest element not yet processed in this column.
        size_t max_row  = k;
        T      max_elem = A.subscript_no_check(k, k);

        for (size_t i = k; i < m; i++) {
            if (std::abs(A.subscript(i, k)) > std::abs(max_elem)) {
                max_row  = i;
                max_elem = A.subscript(i, k);
            }
        }

        // If the current column only has zeros, then skip this iteration.
        if (max_elem == T{0.0}) {
            ret = (int)k;
            continue;
        }

        // Swap the current row and the biggest row.
        pivot[k] = max_row + 1; // Plus 1 to keep it compatible with LAPACK.

        // Checks to see if we actually need to swap.
        if (max_row != k) {
            for (size_t j = 0; j < n; j++) {
                std::swap(A.subscript_no_check(k, j), A.subscript_no_check(max_row, j));
            }
        }

        // Eliminate the rows.
        T row_scale = A.subscript_no_check(k, k);
        for (size_t i = k + 1; i < m; i++) {
            T curr_scale = A.subscript_no_check(i, k);
            for (size_t j = k; j < n; j++) {
                // Do it like this to hopefully avoid over/underflow.
                A.subscript_no_check(i, j) = A.subscript_no_check(i, j) - curr_scale * A.subscript_no_check(k, j) / row_scale;
            }

            // Set the value for the L matrix. The diagonal should be unit.
            A.subscript(i, k) = curr_scale / row_scale;
        }
    }

    pivot[min_dim - 1] = min_dim;

    return ret;
}

template <typename T, Container Pivots>
int impl_solve(einsums::detail::TensorImpl<T> &A, einsums::detail::TensorImpl<T> &X, Pivots &pivot) {
    size_t const m = A.dim(0), n = A.dim(1), min_dim = std::min(m, n), nrhs = X.dim(1);

    // LU decomposition.
    int info = impl_lu_decomp(A, pivot);

    if (info != 0) {
        return info;
    }

    // Solve the system.

    if (X.rank() == 1) {
        // First, apply P^-1 and L^-1.
        for (size_t k = 0; k < min_dim - 1; k++) {
            // We know where the max row was. Swap.
            if (pivot[k] != k + 1) {
                std::swap(X.subscript_no_check(k), X.subscript_no_check(pivot[k] - 1));
            }

            // We know the scale factors for the rows. Apply them
            for (size_t i = k + 1; i < m; i++) {
                T scale = A.subscript_no_check(i, k);
                X.subscript_no_check(i) -= X.subscript(k) * scale;
            }
        }

        // Finally, apply U^-1.
        for (size_t k = n - 1; k > 0; k--) {
            T row_scale = A.subscript_no_check(k, k);
            for (ptrdiff_t i = (ptrdiff_t)k - 1; i >= 0; i--) {
                T scale = A.subscript_no_check(i, k);

                X.subscript_no_check(i) = (row_scale * X.subscript_no_check(i) - scale * X.subscript_no_check(k)) / row_scale;
            }

            X.subscript_no_check(k) /= row_scale;
        }
    } else {

        // First, apply P^-1 and L^-1.
        for (size_t k = 0; k < min_dim - 1; k++) {
            // We know where the max row was. Swap.
            if (pivot[k] != k + 1) {
                for (size_t j = 0; j < nrhs; j++) {
                    std::swap(X.subscript_no_check(k, j), X.subscript_no_check(pivot[k] - 1, j));
                }
            }
        }
        for (size_t k = 0; k < min_dim - 1; k++) {
            // We know the scale factors for the rows. Apply them
            for (size_t i = k + 1; i < m; i++) {
                T scale = A.subscript_no_check(i, k);
                for (size_t j = 0; j < nrhs; j++) {
                    X.subscript_no_check(i, j) -= X.subscript(k, j) * scale;
                }
            }
        }

        // Finally, apply U^-1.
        for (ptrdiff_t k = (ptrdiff_t)n - 1; k >= 0; k--) {
            T row_scale = A.subscript_no_check(k, k);
            for (ptrdiff_t i = (ptrdiff_t)k - 1; i >= 0; i--) {
                T scale = A.subscript_no_check(i, k);

                for (size_t j = 0; j < nrhs; j++) {
                    X.subscript_no_check(i, j) = (row_scale * X.subscript_no_check(i, j) - scale * X.subscript_no_check(k, j)) / row_scale;
                }
            }

            for (size_t j = 0; j < nrhs; j++) {
                X.subscript_no_check(k, j) /= row_scale;
            }
        }
    }

    return info;
}

template <typename T, Container Pivots>
int impl_invert_lu(einsums::detail::TensorImpl<T> &A_lu, Pivots const &pivot) {
    // Assume A_lu has been already put into impl_lu_decomp, and pivot is the result.
    size_t const m = A_lu.dim(0), n = A_lu.dim(1), min_dim = std::min(m, n);

    // First, compute the inverse of U.
    for (size_t k = 0; k < n - 1; k++) {
        // Reduce rows from the current.
        for (size_t i = k + 1; k < n; k++) {
            T scale = -A_lu.subscript_no_check(k, i);

            // Reduce.
            for (size_t j = i + 1; j < n; j++) {
                A_lu.subscript_no_check(k, j) += scale * A_lu.subscript_no_check(i, j);
            }

            // Set the value in A_lu to be the scale.
            A_lu(k, i) = scale;
        }
    }

    // Next, solve for the inverse of A.
}
} // namespace einsums::linear_algebra::detail