#pragma once

#include "einsums/TiledTensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

namespace einsums::linear_algebra::detail {
template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires RankTiledTensor<AType<T, Rank>, Rank, T>;
        requires RankTiledTensor<BType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.grid_size(i) != B.grid_size(i)) {
            throw std::runtime_error("dot: Tiled tensors have incompatible tiles.");
        }

        strides[Rank - 1 - i] = prod;
        prod *= A.grid_size(i);
    }
    T out = 0;

#pragma omp parallel for reduction(+ : out)
    for (size_t index = 0; index < A.grid_size(); index++) {
        std::array<size_t, Rank> index_arr;
        size_t temp_index = index;

        for (int i = 0; i < Rank; i++) {
            index_arr[i] = temp_index / strides[i];
            temp_index %= A.grid_size(i);
        }

        if (!A.has_tile(index_arr) || !B.has_tile(index_arr) || A.has_zero_size(index_arr) || B.has_zero_size(index_arr)) {
            continue;
        }
        out += dot(A.tile(index_arr), B.tile(index_arr));
    }

    return out;
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires RankTiledTensor<AType<T, Rank>, Rank, T>;
        requires RankTiledTensor<BType<T, Rank>, Rank, T>;
    }
auto true_dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.grid_size(i) != B.grid_size(i)) {
            throw std::runtime_error("dot: Tiled tensors have incompatible tiles.");
        }

        strides[Rank - 1 - i] = prod;
        prod *= A.grid_size(i);
    }
    T out = 0;

#pragma omp parallel for reduction(+ : out)
    for (size_t index = 0; index < A.grid_size(); index++) {
        std::array<size_t, Rank> index_arr;
        size_t temp_index = index;

        for (int i = 0; i < Rank; i++) {
            index_arr[i] = temp_index / strides[i];
            temp_index %= A.grid_size(i);
        }

        if (!A.has_tile(index_arr) || !B.has_tile(index_arr) || A.has_zero_size(index_arr) || B.has_zero_size(index_arr)) {
            continue;
        }
        out += true_dot(A.tile(index_arr), B.tile(index_arr));
    }

    return out;
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T, typename U>
    requires requires {
        requires RankTiledTensor<AType<T, Rank>, 2, T>;
        requires RankTiledTensor<BType<T, Rank>, 2, T>;
        requires RankTiledTensor<CType<T, Rank>, 2, T>;
        requires std::convertible_to<U, T>;
    }
void gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const U beta, CType<T, Rank> *C) {
    // Check for compatibility.
    if (C->grid_size(0) != A.grid_size(TransA ? 1 : 0) || C->grid_size(1) != B.grid_size(TransB ? 0 : 1)) {
        throw std::runtime_error("gemm: Output tensor needs to have a compatible tile grid with the inputs.");
    }
    if (A.grid_size(TransA ? 0 : 1) != B.grid_size(TransB ? 1 : 0)) {
        throw std::runtime_error("gemm: Input tensors need to have compatible tile grids.");
    }
    for (int i = 0; i < C->grid_size(0); i++) {
        if (C->tile_size(0)[i] != A.tile_size(TransA ? 1 : 0)[i]) {
            throw std::runtime_error("gemm: Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < C->grid_size(1); i++) {
        if (C->tile_size(1)[i] != B.tile_size(TransB ? 0 : 1)[i]) {
            throw std::runtime_error("gemm: Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < A.grid_size(TransA ? 0 : 1); i++) {
        if (A.tile_size(TransA ? 0 : 1)[i] != B.tile_size(TransB ? 1 : 0)[i]) {
            throw std::runtime_error("gemm: Tile sizes need to match between all three tensors.");
        }
    }
    int x_size = C->grid_size(0), y_size = C->grid_size(1), z_size = A.grid_size(TransA ? 0 : 1);

// For every block in C, do matrix multiplication.
#pragma omp parallel for collapse(2)
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            if (C->has_zero_size(i, j)) {
                continue;
            }
            // Check to see if C will be modified.
            bool modified = false;
            for (int k = 0; k < z_size; k++) {
                if constexpr (TransA && TransB) {
                    modified |= A.has_tile(k, i) && B.has_tile(j, k) && !A.has_zero_size(k, i) && !B.has_zero_size(j, k);
                } else if constexpr (!TransA && TransB) {
                    modified |= A.has_tile(i, k) && B.has_tile(j, k) && !A.has_zero_size(i, k) && !B.has_zero_size(j, k);
                } else if constexpr (TransA && !TransB) {
                    modified |= A.has_tile(k, i) && B.has_tile(k, j) && !A.has_zero_size(k, i) && !B.has_zero_size(k, j);
                } else {
                    modified |= A.has_tile(i, k) && B.has_tile(k, j) && !A.has_zero_size(i, k) && !B.has_zero_size(k, j);
                }
            }

            // If C is modified, then loop through and matrix multiply. Otherwise, scale or delete depending on beta.
            if (modified) {
                C->lock();
                bool  created = !C->has_tile(i, j);
                auto &C_tile  = C->tile(i, j);
                C->unlock();
                if (beta == U{0.0} || created) {
                    C_tile.zero();
                } else {
                    C_tile *= beta;
                }

                for (int k = 0; k < z_size; k++) {
                    if constexpr (TransA && TransB) {
                        if (A.has_tile(k, i) && B.has_tile(j, k) && !A.has_zero_size(k, i) && !B.has_zero_size(j, k)) {
                            gemm<TransA, TransB>(alpha, A.tile(k, i), B.tile(j, k), U{1.0}, &C_tile);
                        }
                    } else if constexpr (TransA && !TransB) {
                        if (A.has_tile(k, i) && B.has_tile(k, j) && !A.has_zero_size(k, i) && !B.has_zero_size(k, j)) {
                            gemm<TransA, TransB>(alpha, A.tile(k, i), B.tile(k, j), U{1.0}, &C_tile);
                        }
                    } else if constexpr (!TransA && TransB) {
                        if (A.has_tile(i, k) && B.has_tile(j, k) && !A.has_zero_size(i, k) && !B.has_zero_size(j, k)) {
                            gemm<TransA, TransB>(alpha, A.tile(i, k), B.tile(j, k), U{1.0}, &C_tile);
                        }
                    } else {
                        if (A.has_tile(i, k) && B.has_tile(k, j) && !A.has_zero_size(i, k) && !B.has_zero_size(k, j)) {
                            gemm<TransA, TransB>(alpha, A.tile(i, k), B.tile(k, j), U{1.0}, &C_tile);
                        }
                    }
                }
            } else {
                C->lock();
                if (C->has_tile(i, j) && !C->has_zero_size(i, j)) {
                    if (beta == U{0.0}) {
                        // C->tiles().erase(std::array<int, 2>{i, j});
                        C->tile(i, j).zero();
                    } else {
                        C->tile(i, j) *= beta;
                    }
                }
                C->unlock();
            }
        }
    }
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T, typename U>
    requires requires {
        requires RankTiledTensor<AType<T, ARank>, 2, T>;
        requires std::convertible_to<U, T>; // Make sure the alpha and beta can be converted to T
    }
void gemv(const U alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const U beta, YType<T, XYRank> *y) {
    if constexpr (einsums::detail::IsTiledTensorV<XType<T, XYRank>>) {
        if constexpr (TransA) {
            if (A.tile_sizes(0) != z.tile_sizes(0)) {
                throw std::runtime_error("gemv: Tiled tensors need to have compatible tile sizes.");
            }
        } else {
            if (A.tile_sizes(1) != z.tile_sizes(0)) {
                throw std::runtime_error("gemv: Tiled tensors need to have compatible tile sizes.");
            }
        }
    }
    if constexpr (einsums::detail::IsTiledTensorV<YType<T, XYRank>>) {
        if constexpr (TransA) {
            if (A.tile_sizes(1) != y->tile_sizes(0)) {
                throw std::runtime_error("gemv: Tiled tensors need to have compatible tile sizes.");
            }
        } else {
            if (A.tile_sizes(0) != y->tile_sizes(0)) {
                throw std::runtime_error("gemv: Tiled tensors need to have compatible tile sizes.");
            }
        }
    }

    if (beta == U(0.0)) {
        y->zero();
    } else {
        *y *= beta;
    }

    int loop_bound = (TransA) ? A.grid_size(1) : A.grid_size(0);
    //        EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < loop_bound; i++) {
        if (A.tile_size((TransA) ? 1 : 0)[i] == 0) {
            continue;
        }
        if constexpr (einsums::detail::IsTiledTensorV<YType<T, XYRank>>) {
            y->lock();
            auto &y_tile = y->tile(i);
            y->unlock();
            for (int j = 0; j < A.grid_size((TransA) ? 0 : 1); j++) {
                if constexpr (TransA) {
                    if (!A.has_tile(j, i) || A.has_zero_size(j, i)) {
                        continue;
                    }
                } else {
                    if (!A.has_tile(i, j) || A.has_zero_size(i, j)) {
                        continue;
                    }
                }
                if constexpr (einsums::detail::IsTiledTensorV<XType<T, XYRank>>) {
                    if constexpr (TransA) {
                        gemv<TransA>(alpha, A.tile(j, i), z.tile(j), U(1.0), &y_tile);
                    } else {
                        gemv<TransA>(alpha, A.tile(i, j), z.tile(j), U(1.0), &y_tile);
                    }
                } else {
                    if constexpr (TransA) {
                        gemv<TransA>(alpha, A.tile(j, i),
                                     z(Range{A.tile_offset((TransA) ? 0 : 1)[j],
                                             A.tile_offset((TransA) ? 0 : 1)[j] + A.tile_size((TransA) ? 0 : 1)[j]}),
                                     U(1.0), &y_tile);
                    } else {
                        gemv<TransA>(alpha, A.tile(i, j),
                                     z(Range{A.tile_offset((TransA) ? 0 : 1)[j],
                                             A.tile_offset((TransA) ? 0 : 1)[j] + A.tile_size((TransA) ? 0 : 1)[j]}),
                                     U(1.0), &y_tile);
                    }
                }
            }
        } else {
            auto y_tile =
                (*y)(Range{A.tile_offset((TransA) ? 1 : 0)[i], A.tile_offset((TransA) ? 1 : 0)[i] + A.tile_size((TransA) ? 1 : 0)[i]});
            for (int j = 0; j < A.grid_size((TransA) ? 0 : 1); j++) {
                if constexpr (TransA) {
                    if (!A.has_tile(j, i) || A.has_zero_size(j, i)) {
                        continue;
                    }
                } else {
                    if (!A.has_tile(i, j) || A.has_zero_size(i, j)) {
                        continue;
                    }
                }
                if constexpr (einsums::detail::IsTiledTensorV<XType<T, XYRank>>) {
                    if constexpr (TransA) {
                        gemv<TransA>(alpha, A.tile(j, i), z.tile(j), U(1.0), &y_tile);
                    } else {
                        gemv<TransA>(alpha, A.tile(i, j), z.tile(j), U(1.0), &y_tile);
                    }
                } else {
                    if constexpr (TransA) {
                        gemv<TransA>(alpha, A.tile(j, i),
                                     z(Range{A.tile_offset((TransA) ? 0 : 1)[j],
                                             A.tile_offset((TransA) ? 0 : 1)[j] + A.tile_size((TransA) ? 0 : 1)[j]}),
                                     U(1.0), &y_tile);
                    } else {
                        gemv<TransA>(alpha, A.tile(i, j),
                                     z(Range{A.tile_offset((TransA) ? 0 : 1)[j],
                                             A.tile_offset((TransA) ? 0 : 1)[j] + A.tile_size((TransA) ? 0 : 1)[j]}),
                                     U(1.0), &y_tile);
                    }
                }
            }
        }
    }
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires RankTiledTensor<AType<T, Rank>, Rank, T>;
        requires RankTiledTensor<BType<T, Rank>, Rank, T>;
        requires RankTiledTensor<CType<T, Rank>, Rank, T>;
    }
void direct_product(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < A.grid_size(); sentinel++) {
        std::array<int, Rank> index = std::array<int, Rank>{};
        size_t                hold  = sentinel;

        // Calculate the index.
        for (int i = 0; i < Rank; i++) {
            index[i] = hold % A.grid_size(i);
            hold /= A.grid_size(i);
        }
        if (A.has_tile(index) && B.has_tile(index)) {
            direct_product(alpha, A.tile(index), B.tile(index), beta, &(C->tile(index)));
        }
    }
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename XYType, size_t XYRank, typename T>
    requires requires {
        requires RankTiledTensor<AType<T, ARank>, 2, T>;
        requires RankTiledTensor<XYType<T, XYRank>, 1, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    if (A->grid_size(0) != X.grid_size(0) && A->grid_size(1) != Y.grid_size(0)) {
        throw std::runtime_error("ger: Tiled tensors have incompatible grids!");
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < X.grid_size(); i++) {
        for (int j = 0; j < Y.grid_size(); j++) {
            if (X.has_zero_size(i) || Y.has_zero_size(j) || !X.has_tile(i) || !Y.has_tile(j)) {
                continue;
            }
            auto &a_tile = A->tile(i, j);

            ger(alpha, X.tile[i], Y.tile[j], &a_tile);
        }
    }
}

template <template <typename, size_t> typename AType, typename T, size_t Rank>
requires RankTiledTensor<AType<T, Rank>, Rank, T>
void scale(T alpha, AType<T, Rank> *A) {
    EINSUMS_OMP_PARALLEL_FOR
    for (auto it = A->tiles().begin(); it != A->tiles().end(); it++) {
        scale(alpha, &(it->second));
    }
}

} // namespace einsums::linear_algebra::detail