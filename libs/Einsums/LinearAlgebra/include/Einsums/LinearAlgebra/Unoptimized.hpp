//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>

#include <cstdint>
#include <Einsums/Config/CompilerSpecific.hpp>

namespace einsums::linear_algebra::detail {

template <TensorConcept AType, TensorConcept BType, typename Result = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>>
    requires requires {
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType>;
        requires SameRank<AType, BType>;
    }
auto dot(AType const &A, BType const &B) -> Result {
    constexpr std::size_t Rank = AType::Rank;

    Result                        out{0.0};
    std::array<std::size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Generic tensors have incompatible dimensions!");
        }
        strides[Rank - i - 1] = prod;

        prod *= A.dim(Rank - i - 1);
    }

#pragma omp parallel for reduction(+ : out) default(none)
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }

        out += (Result)subscript_tensor(A, index) * (Result)subscript_tensor(B, index);
    }

    return out;
}

template <TensorConcept AType, TensorConcept BType, typename Result = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>>
    requires requires {
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType>;
        requires SameRank<AType, BType>;
    }
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    constexpr size_t Rank = AType::Rank;
    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    T                        out{0.0};
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Generic tensors have incompatible dimensions!");
        }
        strides[Rank - i - 1] = prod;

        prod *= A.dim(Rank - i - 1);
    }

#pragma omp parallel for reduction(+ : out) default(none)
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }

        if constexpr (IsComplexV<typename AType::ValueType>) {
            out += (T)std::conj(subscript_tensor(A, index)) * (T)subscript_tensor(B, index);
        } else {
            out += (T)subscript_tensor(A, index) * (T)subscript_tensor(B, index);
        }
    }

    return out;
}

template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U>
    requires requires {
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType> || !AlgebraTensorConcept<CType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    // Check for compatibility.
    if (((TransA) ? A.dim(0) : A.dim(1)) != ((TransB) ? B.dim(1) : B.dim(0))) {
        EINSUMS_THROW_CODED_EXCEPTION(dimension_error, 0, "Matrices require compatible inner dimensions!");
    }
    if (((TransA) ? A.dim(1) : A.dim(0)) != C->dim(0)) {
        EINSUMS_THROW_CODED_EXCEPTION(dimension_error, 1, "Input and output matrices need to have compatible rows!");
    }
    if (((TransB) ? B.dim(0) : B.dim(1)) != C->dim(1)) {
        EINSUMS_THROW_CODED_EXCEPTION(dimension_error, 1, "Input and output matrices need to have compatible columns!");
    }

    size_t rows  = (TransA) ? A.dim(1) : A.dim(0);
    size_t cols  = (TransB) ? B.dim(0) : B.dim(1);
    size_t inner = (TransA) ? A.dim(0) : A.dim(1);

#pragma omp parallel for collapse(2) default(none)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            typename CType::ValueType sum{0.0};
            for (size_t k = 0; k < inner; k++) {

                if constexpr (TransA && TransB) {
                    sum += A(k, i) * B(j, k);
                } else if constexpr (TransA) {
                    sum += A(k, i) * B(k, j);
                } else if constexpr (TransB) {
                    sum += A(i, k) * B(j, k);
                } else {
                    sum += A(i, k) * B(k, j);
                }
            }
            if (beta == U{0.0}) {
                (*C)(i, j) = alpha * sum;
            } else {
                (*C)(i, j) = beta * (*C)(i, j) + alpha * sum;
            }
        }
    }
}

template <bool TransA, MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires std::convertible_to<U, typename AType::ValueType>;
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<XType> || !AlgebraTensorConcept<YType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    // Check bounds.
    if (((TransA) ? A.dim(0) : A.dim(1)) != z.dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Matrix and input vector need to have compatible sizes!");
    }
    if (((TransA) ? A.dim(1) : A.dim(0)) != y->dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Matrix and output vector need to have compatible sizes!");
    }

    size_t rows = y->dim(0);
    size_t cols = z.dim(0);

#pragma omp parallel for default(none)
    for (size_t i = 0; i < rows; i++) {
        typename YType::ValueType sum{0.0};

        for (size_t j = 0; j < cols; j++) {
            if constexpr (TransA) {
                sum += A(j, i) * z(j);
            } else {
                sum += A(i, j) * z(j);
            }
        }

        if (beta == U{0.0}) {
            (*y)(i) = alpha * sum;
        } else {
            (*y)(i) = beta * (*y)(i) + alpha * sum;
        }
    }
}

template <TensorConcept AType, TensorConcept BType, TensorConcept CType, typename U>
    requires requires {
        requires SameRank<AType, BType, CType>;
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType> || !AlgebraTensorConcept<CType>;
    }
void direct_product(U alpha, AType const &A, BType const &B, U, CType *C) {
    using T               = typename AType::ValueType;
    constexpr size_t Rank = AType::Rank;

    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i) || A.dim(i) != C->dim(i)) {
            std::string message;
            if (i % 10 == 1 && (i % 100 > 20 || i % 100 == 1)) {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}st dimensions are A: {}, B: {}, C: {}", i,
                                      A.dim(i), B.dim(i), C->dim(i));
            } else if (i % 10 == 2 && (i % 100 > 20 || i % 100 == 2)) {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}nd dimensions are A: {}, B: {}, C: {}", i,
                                      A.dim(i), B.dim(i), C->dim(i));
            } else if (i % 10 == 3 && (i % 100 > 20 || i % 100 == 3)) {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}rd dimensions are A: {}, B: {}, C: {}", i,
                                      A.dim(i), B.dim(i), C->dim(i));
            } else {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}th dimensions are A: {}, B: {}, C: {}", i,
                                      A.dim(i), B.dim(i), C->dim(i));
            }
            EINSUMS_THROW_EXCEPTION(dimension_error, fmt::runtime(message));
        }
        strides[Rank - i - 1] = prod;

        prod *= A.dim(Rank - i - 1);
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }

        subscript_tensor(*C, index) = alpha * subscript_tensor(A, index) * subscript_tensor(B, index);
    }
}

template <MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires { requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<XType> || !AlgebraTensorConcept<YType>; }
void ger(U alpha, XType const &X, YType const &Y, AType *A) {
    if (A->dim(0) != X.dim(0) || A->dim(1) != Y.dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Incompatible matrix and vector sizes!");
    }

    size_t rows = A->dim(0);
    size_t cols = A->dim(1);

#pragma omp parallel for collapse(2) default(none)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            (*A)(i, j) += alpha * X(i) * Y(j);
        }
    }
}

template <TensorConcept AType>
    requires(!AlgebraTensorConcept<AType>)
void scale(typename AType::ValueType alpha, AType *A) {
    constexpr size_t         Rank = AType::Rank;
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        strides[Rank - i - 1] = prod;

        prod *= A->dim(Rank - i - 1);
    }

#pragma omp parallel for default(none)
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }
        subscript_tensor(*A, index) *= alpha;
    }
}

} // namespace einsums::linear_algebra::detail