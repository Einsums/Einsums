#pragma once

#include "einsums/Blas.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

namespace einsums::linear_algebra::detail {

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
    requires CoreRankBasicTensor<AType<ADataType, ARank>, 1, ADataType>
void sum_square(const AType<ADataType, ARank> &a, RemoveComplexT<ADataType> *scale, RemoveComplexT<ADataType> *sumsq) {
    int n    = a.dim(0);
    int incx = a.stride(0);
    blas::lassq(n, a.data(), incx, scale, sumsq);
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T, typename U>
    requires requires {
        requires CoreRankBasicTensor<AType<T, Rank>, 2, T>;
        requires CoreRankBasicTensor<BType<T, Rank>, 2, T>;
        requires CoreRankBasicTensor<CType<T, Rank>, 2, T>;
        requires std::convertible_to<U, T>;
    }
void gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const U beta, CType<T, Rank> *C) {
    auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
    auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    blas::gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, static_cast<T>(alpha), A.data(), lda, B.data(), ldb, static_cast<T>(beta),
               C->data(), ldc);
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T, typename U>
    requires requires {
        requires CoreRankBasicTensor<AType<T, ARank>, 2, T>;
        requires CoreRankBasicTensor<XType<T, XYRank>, 1, T>;
        requires CoreRankBasicTensor<YType<T, XYRank>, 1, T>;
        requires std::convertible_to<U, T>; // Make sure the alpha and beta can be converted to T
    }
void gemv(const U alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const U beta, YType<T, XYRank> *y) {
    auto m = A.dim(0), n = A.dim(1);
    auto lda  = A.stride(0);
    auto incx = z.stride(0);
    auto incy = y->stride(0);

    blas::gemv(TransA ? 't' : 'n', m, n, static_cast<T>(alpha), A.data(), lda, z.data(), incx, static_cast<T>(beta), y->data(), incy);
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
    requires requires {
        requires CoreRankBasicTensor<AType<T, ARank>, 2, T>;
        requires CoreRankBasicTensor<WType<T, WRank>, 1, T>;
        requires !Complex<T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {
    assert(A->dim(0) == A->dim(1));

    auto           n     = A->dim(0);
    auto           lda   = A->stride(0);
    int            lwork = 3 * n;
    std::vector<T> work(lwork);

    blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
}

template <template <typename, size_t> typename AType, size_t ARank, template <Complex, size_t> typename WType, size_t WRank, typename T,
          bool ComputeLeftRightEigenvectors = true>
    requires requires {
        requires CoreRankBasicTensor<AType<T, ARank>, 2, T>;
        requires CoreRankBasicTensor<WType<AddComplexT<T>, WRank>, 1, AddComplexT<T>>;
    }
void geev(AType<T, ARank> *A, WType<AddComplexT<T>, WRank> *W, AType<T, ARank> *lvecs, AType<T, ARank> *rvecs) {
    assert(A->dim(0) == A->dim(1));
    assert(W->dim(0) == A->dim(0));
    assert(A->dim(0) == lvecs->dim(0));
    assert(A->dim(1) == lvecs->dim(1));
    assert(A->dim(0) == rvecs->dim(0));
    assert(A->dim(1) == rvecs->dim(1));

    blas::geev(ComputeLeftRightEigenvectors ? 'v' : 'n', ComputeLeftRightEigenvectors ? 'v' : 'n', A->dim(0), A->data(), A->stride(0),
               W->data(), lvecs->data(), lvecs->stride(0), rvecs->data(), rvecs->stride(0));
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
    requires requires {
        requires CoreRankBasicTensor<AType<T, ARank>, 2, T>;
        requires CoreRankBasicTensor<WType<RemoveComplexT<T>, WRank>, 1, RemoveComplexT<T>>;
        requires Complex<T>;
    }
void heev(AType<T, ARank> *A, WType<RemoveComplexT<T>, WRank> *W) {
    assert(A->dim(0) == A->dim(1));

    auto                           n     = A->dim(0);
    auto                           lda   = A->stride(0);
    int                            lwork = 2 * n;
    std::vector<T>                 work(lwork);
    std::vector<RemoveComplexT<T>> rwork(3 * n);

    blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires CoreRankBasicTensor<AType<T, ARank>, 2, T>;
        requires CoreRankBasicTensor<BType<T, BRank>, 2, T>;
    }
auto gesv(AType<T, ARank> *A, BType<T, BRank> *B) -> int {
    auto n   = A->dim(0);
    auto lda = A->dim(0);
    auto ldb = B->dim(1);

    auto nrhs = B->dim(0);

    int                   lwork = n;
    std::vector<blas_int> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankBasicTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A) {
    blas::scal(A->dim(0) * A->stride(0), scale, A->data(), 1);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankBasicTensor<AType<T, ARank>, 2, T>
void scale_row(size_t row, T scale, AType<T, ARank> *A) {
    blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankBasicTensor<AType<T, ARank>, 2, T>
void scale_column(size_t col, T scale, AType<T, ARank> *A) {
    blas::scal(A->dim(0), scale, A->data(0ul, col), A->stride(0));
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T>
    requires requires {
        requires CoreRankBasicTensor<AType<T, 1>, 1, T>;
        requires CoreRankBasicTensor<BType<T, 1>, 1, T>;
    }
auto dot(const AType<T, 1> &A, const BType<T, 1> &B) -> T {
    assert(A.dim(0) == B.dim(0));

    auto result = blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    return result;
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires CoreRankBasicTensor<AType<T, Rank>, Rank, T>;
        requires CoreRankBasicTensor<BType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i));
        dim[0] *= A.dim(i);
    }

    return dot(TensorView<T, 1>(const_cast<AType<T, Rank> &>(A), dim), TensorView<T, 1>(const_cast<BType<T, Rank> &>(B), dim));
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires CoreRankBasicTensor<AType<T, Rank>, Rank, T>;
        requires CoreRankBasicTensor<BType<T, Rank>, Rank, T>;
        requires CoreRankBasicTensor<CType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B, const CType<T, Rank> &C) -> T {
    Dim<1> dim{1};

    for (size_t i = 0; i < Rank; i++) {
        assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
        dim[0] *= A.dim(i);
    }

    auto vA = TensorView<T, 1>(const_cast<AType<T, Rank> &>(A), dim);
    auto vB = TensorView<T, 1>(const_cast<BType<T, Rank> &>(B), dim);
    auto vC = TensorView<T, 1>(const_cast<CType<T, Rank> &>(C), dim);

    T result{0};
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < dim[0]; i++) {
        result += vA(i) * vB(i) * vC(i);
    }
    return result;
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires CoreRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires CoreRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    blas::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires CoreRankBasicTensor<XType<T, Rank>, Rank, T>;
        requires CoreRankBasicTensor<YType<T, Rank>, Rank, T>;
    }
void axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y) {
    blas::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires CoreRankTensor<XYType<T, XYRank>, 1, T>;
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    blas::ger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::CoreRankBasicTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::CoreRankBasicTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::CoreRankBasicTensor<CType<T, Rank>, 2, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C) {
    int temp_rows, temp_cols;
    if constexpr (TransA && TransB) {
        assert(B.dim(0) == A.dim(0) && A.dim(1) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else if constexpr (TransA && !TransB) {
        assert(B.dim(1) == A.dim(0) && A.dim(1) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    } else if constexpr (!TransA && TransB) {
        assert(B.dim(0) == A.dim(1) && A.dim(0) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else {
        assert(B.dim(1) == A.dim(1) && A.dim(0) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    }

    if constexpr (TransA) {
        temp_rows = A.dim(1);
    } else {
        temp_rows = A.dim(0);
    }

    if constexpr (TransB) {
        temp_cols = B.dim(0);
    } else {
        temp_cols = B.dim(1);
    }

    *C = T(0.0);

    Tensor<T, 2> temp{"temp", temp_rows, temp_cols};

    gemm<TransA, TransB>(T{1.0}, A, B, T{0.0}, &temp);
    gemm<!TransB, false>(T{1.0}, B, temp, T{0.0}, C);
}

} // namespace einsums::linear_algebra::detail