#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <NotComplex T>
void impl_tridagonal_reduce(einsums::detail::TensorImpl<T> *A, T *work) {
    size_t const dim = A->dim(0), row_stride = A->stride(0), col_stride = A->stride(1);
    T           *vec1 = work, *vec2 = work + dim, *tau = work + 2 * dim;

    T *A_data = A->data();

    for (size_t k = 0; k < dim - 2; k++) {
        T alpha_sq = 0, scale = 0, alpha = 0;

        T key = A->subscript(k + 1, k);

        // Calculate alpha squared and the sign of alpha. The sign is -1 for non-negative values (and zero) and 1 for negative values of the
        // key.
        if (key < 0) {
            alpha = 1;
        } else {
            alpha = -1;
        }
        blas::lassq(dim - k - 1, A->data(k + 1, k), col_stride, &scale, &alpha_sq);

        alpha *= std::sqrt(alpha_sq) * scale;

        // Calculate r.
        T r = std::sqrt((alpha_sq * scale * scale - key * alpha) / 2);

        // Now, we need to calculate the values of the vectors.

        // Don't do this. We never need these elements, so they will be used for tau.
        // for (size_t i = 0; i < k; i++) {
        //     vec1[i] = T{0.0};
        // }

        // Since we never clear the first set of elements, this will set the value for tau on output.
        vec1[k + 1] = (A_data[(k + 1) * row_stride + k * col_stride] - alpha) / (2 * r);

        for (size_t i = k + 2; i < dim; i++) {
            vec1[i] = A_data[i * row_stride + k * col_stride] / (2 * r);
        }

        // Now, we need to calculate Av.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // The first k + 1 elements of vec1 are all zero. This also means we can ignore the first k rows of A.
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] += A_data[i * row_stride + j * col_stride] * vec1[j];
            }
        }

        // Scale by 2.
        blas::scal(dim, T{2.0}, vec2, 1);

        // Next, update A to be A - 2Avv^T.
        // The result of Avv^T is that the first k + 1 elements of vec1 are zero,
        // and the first k elements of vec2 are zero. This means that only the current row will be modified.
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] -= vec2[i] * vec1[j];
            }
        }

        // Now, do it in reverse. Calculate v^T(A - Avv^T)
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // In this case, the first k columns of A - Avv^T should be zero when vec1 is non-zero.
        // This means that the first k entries of vec2 will be zero.
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] += A_data[j * row_stride + i * col_stride] * vec1[j];
            }
        }

        // Scale by 2.
        blas::scal(dim, T{2.0}, vec2, 1);

        // Finally, update A to be (A - 2Avv^T) - 2vv^T(A - 2Avv^T), which is (I - 2vv^T)A(I - 2vv^T).
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] -= vec2[j] * vec1[i];
            }
        }

        // LAPACK: On exit, the diagonal and first super diagonal are overwritten by the corresponding elements.
        // This is by construction, so nothing to do.

        // LAPACK: the elements above the first super diagonal are overwritten by the elements of the elementary reflectors.
        // The way it defines it is that the first non-zero element is 1, so we need to rescale the reflectors.

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = k + 2; i < dim; i++) {
            A_data[i * row_stride + k * col_stride] = vec1[i] / vec1[k + 1];
        }

        // Set tau.
        tau[k] = 2 * vec1[k + 1] * vec1[k + 1];
    }

    tau[dim - 2] = T{1.0};
}

template <Complex T>
void impl_tridagonal_reduce(einsums::detail::TensorImpl<T> *A, T *work) {
    size_t const dim = A->dim(0), row_stride = A->stride(0), col_stride = A->stride(1);
    T           *vec1 = work, *vec2 = work + dim, *tau = work + 2 * dim;

    T *A_data = A->data();

    for (size_t k = 0; k < dim - 2; k++) {
        RemoveComplexT<T> alpha_sq = 0.0, scale = 0.0, alpha = 0.0;

        T key = A->subscript(k + 1, k);

        // Calculate alpha squared and the sign of alpha. The sign is -1 for non-negative values (and zero) and 1 for negative values of the
        // key.
        blas::lassq(dim - k - 1, A->data(k + 1, k), col_stride, &scale, &alpha_sq);

        alpha *= std::sqrt(alpha_sq) * scale;

        // Calculate tau.
        RemoveComplexT<T> r = std::sqrt(alpha_sq * scale * scale - std::abs(key) * alpha);
        tau[k]              = -((std::conj(key) + alpha) / (key + alpha) + T{1.0}) / (r * r);

        // Now, we need to calculate the values of the vectors.

        // Don't do this. We never need these elements, so they will be used for tau.
        // for (size_t i = 0; i < k; i++) {
        //     vec1[i] = T{0.0};
        // }

        // Since we never clear the first set of elements, this will set the value for tau on output.
        vec1[k + 1] = (A_data[(k + 1) * row_stride + k * col_stride] - alpha) / r;

        for (size_t i = k + 2; i < dim; i++) {
            vec1[i] = A_data[i * row_stride + k * col_stride] / r;
        }

        // Now, we need to calculate Av.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // The first k + 1 elements of vec1 are all zero. This also means we can ignore the first k rows of A.
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] += A_data[i * row_stride + j * col_stride] * vec1[j];
            }
        }

        // Scale by the scale factor.
        blas::scal(dim, std::conj(tau[k]), vec2, 1);

        // Next, update A to be A - tau^* Avv^H.
        // The result of Avv^H is that the first k + 1 elements of vec1 are zero,
        // and the first k elements of vec2 are zero. This means that only the current row will be modified.
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] -= vec2[i] * std::conj(vec1[j]);
            }
        }

        // Now, do it in reverse. Calculate v^H(A - Avv^H)
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // In this case, the first k columns of A - tau^* Avv^H should be zero when vec1 is non-zero.
        // This means that the first k entries of vec2 will be zero.
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] += A_data[j * row_stride + i * col_stride] * std::conj(vec1[j]);
            }
        }

        // Scale by the scale factor.
        blas::scal(dim, tau[k], vec2, 1);

        // Finally, update A to be (A - 2Avv^T) - 2vv^T(A - 2Avv^T), which is (I - 2vv^T)A(I - 2vv^T).
        EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] -= vec2[j] * vec1[i];
            }
        }

        // LAPACK: On exit, the diagonal and first super diagonal are overwritten by the corresponding elements.
        // This is by construction, so nothing to do.

        // LAPACK: the elements above the first super diagonal are overwritten by the elements of the elementary reflectors.
        // The way it defines it is that the first non-zero element is 1, so we need to rescale the reflectors.

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = k + 2; i < dim; i++) {
            A_data[i * row_stride + k * col_stride] = vec1[i] / vec1[k + 1];
        }

        // Set tau.
        tau[k] *= vec1[k + 1] * vec1[k + 1];
    }

    tau[dim - 2] = T{1.0};
}

template <typename T>
void impl_compute_q(einsums::detail::TensorImpl<T> *Q, T *work, T *tau) {
#pragma omp parallel
    {
#pragma omp single
        {
            T           *Q_data = Q->data();
            size_t const dim = Q->dim(0), row_stride = Q->stride(0), col_stride = Q->stride(1);
            T           *vec1 = work, *vec2 = work + dim;

            // Set up the first level.
            Q->subscript(-1, -1) = T{1.0};

            // Loop.
            for (int n = dim - 2; n >= 0; n--) {
                // Set up the vector.
                EINSUMS_OMP_PARALLEL_FOR_SIMD
                for (int k = 0; k < n; k++) {
                    vec1[k] = T{0.0};
                }

                // Put a 1 in the next element.
                vec1[n] = T{1.0};

                // Fill the rest of the elements.
                EINSUMS_OMP_PARALLEL_FOR_SIMD
                for (int k = n + 1; k < dim; k++) {
                    vec1[k] = Q_data[k * row_stride + n * col_stride];
                }

// Now, set up the next level of the matrix.
#pragma omp task
                {
                    EINSUMS_OMP_PARALLEL_FOR_SIMD
                    for (int k = n + 1; k < dim; k++) {
                        Q_data[k * row_stride + n * col_stride] = T{0.0};
                    }
                }
#pragma omp task
                {
                    EINSUMS_OMP_PARALLEL_FOR_SIMD
                    for (int k = n + 1; k < dim; k++) {
                        Q_data[n * row_stride + k * col_stride] = T{0.0};
                    }
                }
#pragma omp task
                {
                    Q_data[n * (row_stride + col_stride)] = T{1.0};
                }
#pragma omp taskgroup

                // Next, find Qv.
                EINSUMS_OMP_PARALLEL_FOR_SIMD
                for (int i = 0; i < dim; i++) {
                    vec2[i] = T{0.0};
                }

                EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
                for (int i = n; i < dim; i++) {
                    for (int j = n; j < dim; j++) {
                        vec2[i] += Q_data[i * row_stride + j * col_stride] * vec1[j];
                    }
                }

                // Next, find Q - Qvv^H.
                if constexpr (IsComplexV<T>) {
                    EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
                    for (int i = n; i < dim; i++) {
                        for (int j = n; j < dim; j++) {
                            // Might need conj(tau).
                            Q_data[i * row_stride + j * col_stride] -= tau[n] * vec2[i] * std::conj(vec1[j]);
                        }
                    }
                } else {
                EINSUMS_OMP_SIMD_PRAGMA(parallel for collapse(2))
                for (int i = n; i < dim; i++) {
                    for (int j = n; j < dim; j++) {
                        Q_data[i * row_stride + j * col_stride] -= tau[n] * vec2[i] * vec1[j];
                    }
                }
                }
            }
        }
    }
}

template <typename T>
void givens_params(T x, T y, T *s, T *c, T *r) {
    if (y == T{0.0}) {
        if (x == T{0.0}) {
            *c = T{1.0};
        } else {
            *c = std::copysign(T{1.0}, x);
        }
        *s = T{0.0};
        *r = std::abs(x);
    } else if (x == T{0.0}) {
        *c = 0;
        *s = std::copysign(T{1.0}, y);
        *r = std::abs(y);
    } else if (std::abs(x) > std::abs(y)) {
        T t = y / x;
        T u = std::copysign(std::sqrt(1 + t * t), x);
        *c  = T{1.0} / u;
        *s  = -*c * t;
        *r  = y * u;
    } else {
        T t = x / y;
        T u = std::copysign(std::sqrt(1 + t * t), y);
        *s  = T{-1.0} / u;
        *c  = t / u;
        *r  = y * u;
    }
}

template <typename T>
void impl_compute_2by2_eigen(T *Q, size_t row_stride, size_t col_stride, RemoveComplexT<T> *diag, RemoveComplexT<T> *subdiag) {
    // Truly diagonal matrix.
    if (*subdiag == RemoveComplexT<T>{0.0}) {
        // Check if we need to swap the eigenvalues.
        if (diag[0] > diag[1]) {
            std::swap(diag[0], diag[1]);
            std::swap(*Q, Q[col_stride]);
            std::swap(Q[row_stride], Q[row_stride + col_stride]);
        }
        // Otherwise, we don't have to do anything. We can just return here.
        return;
    }

    // Set up as a polynomial.
    RemoveComplexT<T> const b = diag[0] + diag[1], c = diag[0] * diag[1] - subdiag[0] * subdiag[0];

    // Find the eigenvalues using the quadratic formula.
    RemoveComplexT<T> const l1 = (-b - std::sqrt(b * b - 4 * c)) / 2;
    RemoveComplexT<T> const l2 = (-b + std::sqrt(b * b - 4 * c)) / 2;

    // Now, find the eigenvectors.
    RemoveComplexT<T> v1x = RemoveComplexT<T>{1.0}, v1y = (l1 - diag[0]) / subdiag[0], v2x = RemoveComplexT<T>{1.0},
                      v2y = (l2 - diag[0]) / subdiag[0];

    // Normalize.
    RemoveComplexT<T> v1norm = std::hypot(v1x, v1y), v2norm = std::hypot(v2x, v2y);
    v1x /= v1norm;
    v1y /= v1norm;
    v2x /= v2norm;
    v2y /= v2norm;

    // Now, we can multiply Q on the left by this.
    T temp11 = v1x * Q[0] + v2x * Q[row_stride], temp12 = v1x * Q[col_stride] + v2x * Q[row_stride + col_stride],
      temp21 = v1y * Q[0] + v2y * Q[row_stride], temp22 = v1y * Q[col_stride] + v2y * Q[row_stride + col_stride];

    // And finally, set the eigenvectors.
    Q[0]                       = temp11;
    Q[row_stride]              = temp21;
    Q[col_stride]              = temp12;
    Q[row_stride + col_stride] = temp22;

    return;
}

template <typename T>
void impl_qr_tridiag_eigen_step(size_t dim, T *Q, size_t Q_rows, size_t row_stride, size_t col_stride, RemoveComplexT<T> *diag,
                                RemoveComplexT<T> *subdiag) {

    // If the matrix is not 2x2, then do the shifted qr algorithm.
    // Start by computing the shifts.
    RemoveComplexT<T> const b = diag[dim - 2] + diag[dim - 1], c = diag[dim - 2] * diag[dim - 1] - subdiag[dim - 2] * subdiag[dim - 2];

    RemoveComplexT<T> shift, s1 = (-b + std::sqrt(b * b - 4 * c)) / 2, s2 = (-b - std::sqrt(b * b - 4 * c)) / 2;

    if (std::abs(diag[dim - 1] - s1) < std::abs(diag[dim - 1] - s2)) {
        shift = s1;
    } else {
        shift = s2;
    }

    // Shift the diagonal elements.

    for (size_t i = 0; i < dim; i++) {
        diag[i] -= shift;
    }

    RemoveComplexT<T> x = diag[0], y = subdiag[0];

    // Use Givens rotations to compute the next step.
    for (size_t i = 0; i < dim - 1; i++) {
        RemoveComplexT<T> r, s, c;
        givens_params(x, y, &s, &c, &r);

        RemoveComplexT<T> w = c * x - s * y;
        RemoveComplexT<T> d = diag[i] - diag[i + 1];
        RemoveComplexT<T> z = (2 * c * subdiag[i] + d * s) * s;
        diag[i] -= z;
        diag[i + 1] += z;
        subdiag[i] = d * c * s + (c * c - s * s) * subdiag[i];

        x = subdiag[i];

        if (i > 0) {
            subdiag[i - 1] = w;
        }

        if (i < dim - 2) {
            y = -s * subdiag[i + 1];
            subdiag[i + 1] *= c;
        }

        // Operate on the eigenvectors.
        for (size_t j = 0; j < Q_rows; j++) {
            T A = Q[j * row_stride + i * col_stride], B = Q[j * row_stride + (i + 1) * col_stride];
            Q[j * row_stride + i * col_stride]       = A * c - B * s;
            Q[j * row_stride + (i + 1) * col_stride] = A * s + B * c;
        }
    }

    for (size_t i = 0; i < dim; i++) {
        diag[i] += shift;
    }

    EINSUMS_LOG_INFO("Diagonal elements:");
    for (int i = 0; i < dim; i++) {
        EINSUMS_LOG_INFO("{}", diag[i]);
    }

    EINSUMS_LOG_INFO("Off-diagonal elements:");
    for (int i = 0; i < dim - 1; i++) {
        EINSUMS_LOG_INFO("{}", subdiag[i]);
    }
}

template <typename T>
void impl_qr_tridiag_iterate(size_t dim, T *Q, size_t Q_rows, size_t row_stride, size_t col_stride, RemoveComplexT<T> *diag,
                             RemoveComplexT<T> *subdiag) {
    // First, check to see if the matrix is 2x2 or 1x1.

    if (dim == 1) {
        // Don't do anything.
        return;
    } else if (dim == 2) {
        T new_vec[4];

        impl_compute_2by2_eigen(new_vec, 2, 1, diag, subdiag);

        // Now, multiply the eigenvectors by the new vector.
        for (size_t i = 0; i < Q_rows; i++) {
            T x = Q[i * row_stride], y = Q[i * row_stride + col_stride];
            Q[i * row_stride]              = x * new_vec[0] + y * new_vec[2];
            Q[i * row_stride + col_stride] = x * new_vec[1] + y * new_vec[3];
        }
        return;
    }

    RemoveComplexT<T> constexpr eps = std::numeric_limits<RemoveComplexT<T>>::epsilon();

    // Now, do the iteration.
    for (int iter = 0; iter < 30 * dim; iter++) {

        size_t split = dim - 1;

        // Now, we can iterate. First, start by finding small values in the subdiagonal.
        for (size_t i = 0; i < dim - 1; i++) {
            if (std::abs(subdiag[i]) <= std::sqrt(std::abs(diag[i]) * std::abs(diag[i + 1])) * eps) {
                subdiag[i] = RemoveComplexT<T>{0.0};
                split      = i;
                break;
            }
            if (std::abs(subdiag[i]) == RemoveComplexT<T>{0.0}) {
                split = i;
                break;
            }
        }

        if (split == dim - 1) {
            // If the split is at the end, then do the QR algorithm.
            impl_qr_tridiag_eigen_step(dim, Q, Q_rows, row_stride, col_stride, diag, subdiag);
        } else {
            // Otherwise, split and reiterate.
            impl_qr_tridiag_iterate(split + 1, Q, Q_rows, row_stride, col_stride, diag, subdiag);
            impl_qr_tridiag_iterate(dim - split - 1, Q + (split + 1) * col_stride, Q_rows, row_stride, col_stride, diag + split + 1,
                                    subdiag + split + 1);
            return;
        }
    }

    EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not converge QR iterations for syev/heev!");
}

template <NotComplex T>
void impl_strided_syev(char jobz, einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<T> *W, T *work) {
    size_t const row_stride = A->stride(0), col_stride = A->stride(1), dim = A->dim(0);
    T           *A_data = A->data();

    // Reduce to tridiagonal form.
    impl_tridagonal_reduce(A, work);

    // Compute the eigenvalues and eigenvectors of the tridiagonal form.
    if (std::tolower(jobz) == 'n') {
        T *diagonal = work + dim, *subdiagonal = work + 2 * dim;

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            diagonal[i] = A_data[i * (row_stride + col_stride)];
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim - 1; i++) {
            subdiagonal[i] = A_data[i * (row_stride + col_stride) + row_stride];
        }

        // Just the eigenvalues. We can use LAPACK's sterf to compute the eigenvalues of a tridiagonal matrix.
        auto info = blas::sterf(A->dim(0), work, work + A->dim(0));

        if (info == -1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The length argument to sterf was invalid! It must be non-negative, got {}.",
                                    A->dim(0));
        } else if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "The {} argument to sterf was invalid! This is likely due to being passed a null pointer.",
                                    print::ordinal(-info));
        } else if (info > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "The algorithm failed to find all of the eigenvalues within {} iterations (30 times the dimension). A "
                                    "total of {} elements have not converged to zero.",
                                    30 * A->dim(0), info);
        }

        // Copy the eigenvalues.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < A->dim(0); i++) {
            W->subscript(i) = work[i];
        }
    } else {
        BufferVector<T> diag(A->dim(0)), subdiag(A->dim(0) - 1);

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            diag[i] = A_data[i * (row_stride + col_stride)];
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim - 1; i++) {
            subdiag[i] = A_data[i * (row_stride + col_stride) + row_stride];
        }

        // Compute the Q matrix.
        impl_compute_q(A, work, work + 2 * A->dim(0));

        impl_qr_tridiag_iterate(A->dim(0), A->data(), A->dim(0), A->stride(0), A->stride(1), diag.data(), subdiag.data());

        for (size_t i = 0; i < dim; i++) {
            W->subscript(i) = diag[i];
        }
    }
}

template <Complex T>
void impl_strided_heev(char jobz, einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<RemoveComplexT<T>> *W, T *work,
                       RemoveComplexT<T> *rwork) {
    size_t const row_stride = A->stride(0), col_stride = A->stride(1), dim = A->dim(0);
    T           *A_data = A->data();

    RemoveComplexT<T> *diag = rwork, *subdiag = rwork + dim;

    // Reduce to tridiagonal form.
    impl_tridagonal_reduce(A, work);

    // Compute the eigenvalues and eigenvectors of the tridiagonal form.
    if (std::tolower(jobz) == 'n') {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            diag[i] = std::real(A_data[i * (row_stride + col_stride)]);
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim - 1; i++) {
            subdiag[i] = std::real(A_data[i * (row_stride + col_stride) + row_stride]);
        }

        // Just the eigenvalues. We can use LAPACK's sterf to compute the eigenvalues of a tridiagonal matrix.
        auto info = blas::sterf(A->dim(0), diag, subdiag);

        if (info == -1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The length argument to sterf was invalid! It must be non-negative, got {}.",
                                    A->dim(0));
        } else if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "The {} argument to sterf was invalid! This is likely due to being passed a null pointer.",
                                    print::ordinal(-info));
        } else if (info > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "The algorithm failed to find all of the eigenvalues within {} iterations (30 times the dimension). A "
                                    "total of {} elements have not converged to zero.",
                                    30 * A->dim(0), info);
        }

        // Copy the eigenvalues.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < A->dim(0); i++) {
            W->subscript(i) = rwork[i];
        }
    } else {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            diag[i] = std::real(A_data[i * (row_stride + col_stride)]);
        }

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim - 1; i++) {
            subdiag[i] = std::real(A_data[i * (row_stride + col_stride) + row_stride]);
        }

        // Compute the Q matrix.
        impl_compute_q(A, work, work + 2 * A->dim(0));

        impl_qr_tridiag_iterate(A->dim(0), A->data(), A->dim(0), A->stride(0), A->stride(1), diag, subdiag);

        for (size_t i = 0; i < dim; i++) {
            W->subscript(i) = diag[i];
        }
    }
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums