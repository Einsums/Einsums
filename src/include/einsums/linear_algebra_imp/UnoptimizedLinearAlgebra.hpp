#pragma once

#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

namespace einsums::linear_algebra::detail {

template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType>;
        requires SameRank<AType, BType>;
    }
auto dot(const AType &A, const BType &B) -> BiggestTypeT<typename AType::data_type, typename BType::data_type> {
    constexpr size_t Rank = AType::Rank;
    using T               = BiggestTypeT<typename AType::data_type, typename BType::data_type>;

    T                        out{0.0};
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i)) {
            throw EINSUMSEXCEPTION("Generic tensors have incompatible dimensions!");
        }
        strides[Rank - i - 1] = prod;

        prod *= A.dim(Rank - i - 1);
    }

#pragma omp parallel for reduction(+ : out)
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }

        out += (T)std::apply(A, index) * (T)std::apply(B, index);
    }

    return out;
}

template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType>;
        requires SameRank<AType, BType>;
    }
auto true_dot(const AType &A, const BType &B) -> typename AType::data_type {
    constexpr size_t Rank = AType::Rank;
    using T               = BiggestTypeT<typename AType::data_type, typename BType::data_type>;

    T                        out{0.0};
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i)) {
            throw EINSUMSEXCEPTION("Generic tensors have incompatible dimensions!");
        }
        strides[Rank - i - 1] = prod;

        prod *= A.dim(Rank - i - 1);
    }

#pragma omp parallel for reduction(+ : out)
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }

        if constexpr (IsComplexV<typename AType::data_type>) {
            out += (T)std::conj(std::apply(A, index)) * (T)std::apply(B, index);
        } else {
            out += (T)std::apply(A, index) * (T)std::apply(B, index);
        }
    }

    return out;
}

template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U>
    requires requires {
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<BType> || !AlgebraTensorConcept<CType>;
        requires std::convertible_to<U, typename AType::data_type>;
    }
void gemm(const U alpha, const AType &A, const BType &B, const U beta, CType *C) {
    // Check for compatibility.
    if (((TransA) ? A.dim(0) : A.dim(1)) != ((TransB) ? B.dim(1) : B.dim(0))) {
        throw EINSUMSEXCEPTION("Matrices require compatible inner dimensions!");
    }
    if (((TransA) ? A.dim(1) : A.dim(0)) != C->dim(0)) {
        throw EINSUMSEXCEPTION("Input and output matrices need to have compatible rows!");
    }
    if (((TransB) ? B.dim(0) : B.dim(1)) != C->dim(1)) {
        throw EINSUMSEXCEPTION("Input and output matrices need to have compatible columns!");
    }

    size_t rows  = (TransA) ? A.dim(1) : A.dim(0);
    size_t cols  = (TransB) ? B.dim(0) : B.dim(1);
    size_t inner = (TransA) ? A.dim(0) : A.dim(1);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            typename CType::data_type sum{0.0};
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
        requires std::convertible_to<U, typename AType::data_type>;
        requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<XType> || !AlgebraTensorConcept<YType>;
    }
void gemv(const U alpha, const AType &A, const XType &z, const U beta, YType *y) {
    // Check bounds.
    if (((TransA) ? A.dim(0) : A.dim(1)) != z.dim(0)) {
        throw EINSUMSEXCEPTION("Matrix and input vector need to have compatible sizes!");
    }
    if (((TransA) ? A.dim(1) : A.dim(0)) != y->dim(0)) {
        throw EINSUMSEXCEPTION("Matrix and output vector need to have compatible sizes!");
    }

    size_t rows = y->dim(0);
    size_t cols = z.dim(0);

#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        typename YType::data_type sum{0.0};

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
void direct_product(U alpha, const AType &A, const BType &B, U, CType *C) {
    using T               = typename AType::data_type;
    constexpr size_t Rank = AType::Rank;

    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.dim(i) != B.dim(i) || A.dim(i) != C->dim(i)) {
            std::string message;
            if(i % 10 == 1 && (i % 100 > 20 || i % 100 == 1)) {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}st dimensions are A: {}, B: {}, C: {}", i, A.dim(i), B.dim(i), C->dim(i));
            } else if(i % 10 == 2 && (i % 100 > 20 || i % 100 == 2)) {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}nd dimensions are A: {}, B: {}, C: {}", i, A.dim(i), B.dim(i), C->dim(i));
            } else if(i % 10 == 3 && (i % 100 > 20 || i % 100 == 3)) {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}rd dimensions are A: {}, B: {}, C: {}", i, A.dim(i), B.dim(i), C->dim(i));
            } else {
                message = fmt::format("Generic tensors have incompatible dimensions! The {}th dimensions are A: {}, B: {}, C: {}", i, A.dim(i), B.dim(i), C->dim(i));
            }
            throw EINSUMSEXCEPTION(message);
        }
        strides[Rank - i - 1] = prod;

        prod *= A.dim(Rank - i - 1);
    }

#pragma omp parallel for
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }

        std::apply(*C, index) = alpha * std::apply(A, index) * std::apply(B, index);
    }
}

template <MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires { requires !AlgebraTensorConcept<AType> || !AlgebraTensorConcept<XType> || !AlgebraTensorConcept<YType>; }
void ger(U alpha, const XType &X, const YType &Y, AType *A) {
    if (A->dim(0) != X.dim(0) || A->dim(1) != Y.dim(0)) {
        throw EINSUMSEXCEPTION("Incompatible matrix and vector sizes!");
    }

    size_t rows = A->dim(0);
    size_t cols = A->dim(1);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            (*A)(i, j) += alpha * X(i) * Y(j);
        }
    }
}

template <TensorConcept AType>
    requires(!AlgebraTensorConcept<AType>)
void scale(typename AType::data_type alpha, AType *A) {
    constexpr size_t         Rank = AType::Rank;
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        strides[Rank - i - 1] = prod;

        prod *= A->dim(Rank - i - 1);
    }

#pragma omp parallel for
    for (size_t sentinel = 0; sentinel < prod; sentinel++) {
        thread_local std::array<int, Rank> index;
        size_t                             quotient = sentinel;

        for (int i = 0; i < Rank; i++) {
            index[i] = quotient / strides[i];
            quotient %= strides[i];
        }
        std::apply(*A, index) *= alpha;
    }
}

} // namespace einsums::linear_algebra::detail