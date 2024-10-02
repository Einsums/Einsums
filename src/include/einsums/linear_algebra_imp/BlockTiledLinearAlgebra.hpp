#pragma once

#include "einsums/BlockTensor.hpp"
#include "einsums/TiledTensor.hpp"
#include "einsums/linear_algebra_imp/BaseLinearAlgebra.hpp"
#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

namespace einsums::linear_algebra::detail {

template <CollectedTensorConcept AType, CollectedTensorConcept BType, CollectedTensorConcept CType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires(TiledTensorConcept<AType> || TiledTensorConcept<BType> || TiledTensorConcept<CType>);
        requires(BlockTensorConcept<AType> || BlockTensorConcept<BType> || BlockTensorConcept<CType>);
        requires !(TiledTensorConcept<AType> && TiledTensorConcept<BType> && BlockTensorConcept<CType>);
    }
void direct_product(typename AType::data_type alpha, const AType &A, const BType &B, typename CType::data_type beta, CType *C) {
    size_t num_blocks;
    if constexpr (einsums::detail::IsBlockTensorV<AType>) {
        num_blocks = A.num_blocks();
    } else {
        num_blocks = B.num_blocks();
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < num_blocks; i++) {
        std::array<int, AType::Rank> index = std::array<int, AType::Rank>{i};

        if constexpr (einsums::detail::IsTiledTensorV<AType> && einsums::detail::IsBlockTensorV<BType> &&
                      einsums::detail::IsTiledTensorV<CType>) {
            if (!A.has_tile(index) || B.block_dim(i) == 0) {
                continue;
            }
            C->lock();
            auto &C_tile = C->tile(index);
            C->unlock();
            direct_product(alpha, A.tile(index), B.block(i), beta, &C_tile);
        } else if constexpr (einsums::detail::IsTiledTensorV<AType> && einsums::detail::IsBlockTensorV<BType> &&
                             einsums::detail::IsBlockTensorV<CType>) {
            if (!A.has_tile(index) || B.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.tile(index), B.block(i), beta, &(C->block(i)));
        } else if constexpr (einsums::detail::IsBlockTensorV<AType> && einsums::detail::IsBlockTensorV<BType> &&
                             einsums::detail::IsTiledTensorV<CType>) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            C->lock();
            auto &C_tile = C->tile(index);
            C->unlock();
            direct_product(alpha, A.block(i), B.block(i), beta, &C_tile);
        } else if constexpr (einsums::detail::IsBlockTensorV<AType> && einsums::detail::IsTiledTensorV<BType> &&
                             einsums::detail::IsTiledTensorV<CType>) {
            if (!B.has_tile(index) || A.block_dim(i) == 0) {
                continue;
            }
            C->lock();
            auto &C_tile = C->tile(index);
            C->unlock();
            direct_product(alpha, A.block(i), B.tile(index), beta, &C_tile);
        } else {
            if (!B.has_tile(index) || A.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.block(i), B.tile(index), beta, &(C->block(i)));
        }
    }
}

template <CollectedTensorConcept AType, CollectedTensorConcept BType>
    requires requires {
        requires(TiledTensorConcept<AType> || TiledTensorConcept<BType>);
        requires(BlockTensorConcept<AType> || BlockTensorConcept<BType>);
        requires SameUnderlyingAndRank<AType, BType>;
    }
auto dot(const AType &A, const BType &B) -> typename AType::data_type {
    for (int i = 0; i < AType::Rank; i++) {
        if (A.grid_size(i) != B.num_blocks()) {
            throw EINSUMSEXCEPTION("Tiled tensor and block tensor have incompatible layouts.");
        }
    }
    using T               = typename AType::data_type;
    constexpr size_t Rank = AType::Rank;
    T                out  = 0;

    int num_blocks;

    if constexpr (einsums::detail::IsBlockTensorV<AType>) {
        num_blocks = A.num_blocks();
    } else {
        num_blocks = B.num_blocks();
    }

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < num_blocks; i++) {
        std::array<size_t, Rank> tile_index;
        for (int j = 0; j < Rank; j++) {
            tile_index[j] = i;
        }
        if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            if (A.block_dim(i) == 0 || !B.has_block(tile_index) || B.has_zero_size(tile_index)) {
                continue;
            }
            out += dot(A[i], B.tile(tile_index));
        } else {
            if (B.block_dim(i) == 0 || !A.has_block(tile_index) || A.has_zero_size(tile_index)) {
                continue;
            }
            out += dot(B[i], A.tile(tile_index));
        }
    }

    return out;
}

template <CollectedTensorConcept AType, CollectedTensorConcept BType>
    requires requires {
        requires(TiledTensorConcept<AType> || TiledTensorConcept<BType>);
        requires(BlockTensorConcept<AType> || BlockTensorConcept<BType>);
        requires SameUnderlyingAndRank<AType, BType>;
    }
auto true_dot(const AType &A, const BType &B) -> typename AType::data_type {
    for (int i = 0; i < AType::Rank; i++) {
        if (A.grid_size(i) != B.num_blocks()) {
            throw EINSUMSEXCEPTION("Tiled tensor and block tensor have incompatible layouts.");
        }
    }
    using T               = typename AType::data_type;
    constexpr size_t Rank = AType::Rank;
    T                out  = T{0.0};

    int num_blocks;

    if constexpr (einsums::detail::IsBlockTensorV<AType>) {
        num_blocks = A.num_blocks();
    } else {
        num_blocks = B.num_blocks();
    }

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < num_blocks; i++) {
        std::array<size_t, Rank> tile_index;
        for (int j = 0; j < Rank; j++) {
            tile_index[j] = i;
        }
        if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            if (A.block_dim(i) == 0 || !B.has_block(tile_index) || B.has_zero_size(tile_index)) {
                continue;
            }
            out += true_dot(A[i], B.tile(tile_index));
        } else {
            if (B.block_dim(i) == 0 || !A.has_block(tile_index) || A.has_zero_size(tile_index)) {
                continue;
            }
            out += true_dot(B[i], A.tile(tile_index));
        }
    }

    return out;
}

template <bool TransA, bool TransB, BlockTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::convertible_to<U, typename AType::data_type>;
    }
void gemm(const U alpha, const AType &A, const BType &B, const U beta, CType *C) {
    // Check for compatibility.
    if (C->grid_size(0) != A.num_blocks() || C->grid_size(1) != B.grid_size(TransB ? 0 : 1)) {
        throw EINSUMSEXCEPTION("Output tensor needs to have a compatible tile grid with the inputs.");
    }
    if (A.num_blocks() != B.grid_size(TransB ? 1 : 0)) {
        throw EINSUMSEXCEPTION("Input tensors need to have compatible tile grids.");
    }
    for (int i = 0; i < C->grid_size(0); i++) {
        if (C->tile_size(0)[i] != A.block_dims(i)) {
            throw EINSUMSEXCEPTION("Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < C->grid_size(1); i++) {
        if (C->tile_size(1)[i] != B.tile_size(TransB ? 0 : 1)[i]) {
            throw EINSUMSEXCEPTION("Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) != B.tile_size(TransB ? 1 : 0)[i]) {
            throw EINSUMSEXCEPTION("Tile sizes need to match between all three tensors.");
        }
    }
    int x_size = C->grid_size(0), y_size = C->grid_size(1);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            if constexpr (TransB) {
                if (C->has_zero_size(i, j) || !B.has_tile(j, i) || B.has_zero_size(j, i) || A.block_dim(i) == 0) {
                    continue;
                }
                C->lock();
                auto &C_tile = C->tile(i, j);
                C->unlock();
                gemm<TransA, TransB>(alpha, A[i], B.tile(j, i), beta, &C_tile);
            } else {
                if (C->has_zero_size(i, j) || !B.has_tile(i, j) || B.has_zero_size(i, j) || A.block_dim(i) == 0) {
                    continue;
                }
                C->lock();
                auto &C_tile = C->tile(i, j);
                C->unlock();
                gemm<TransA, TransB>(alpha, A[i], B.tile(i, j), beta, &C_tile);
            }
        }
    }
}

template <bool TransA, bool TransB, TiledTensorConcept AType, BlockTensorConcept BType, TiledTensorConcept CType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::convertible_to<U, typename AType::data_type>;
    }
void gemm(const U alpha, const AType &A, const BType &B, const U beta, CType *C) {
    // Check for compatibility.
    if (C->grid_size(0) != A.grid_size(TransA ? 1 : 0) || C->grid_size(1) != B.num_blocks()) {
        throw EINSUMSEXCEPTION("Output tensor needs to have a compatible tile grid with the inputs.");
    }
    if (A.grid_size(TransA ? 0 : 1) != B.num_blocks()) {
        throw EINSUMSEXCEPTION("Input tensors need to have compatible tile grids.");
    }
    for (int i = 0; i < C->grid_size(0); i++) {
        if (C->tile_size(0)[i] != A.tile_size(TransA ? 1 : 0)[i]) {
            throw EINSUMSEXCEPTION("Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < C->grid_size(1); i++) {
        if (C->tile_size(1)[i] != B.block_dim(i)) {
            throw EINSUMSEXCEPTION("Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < A.num_blocks; i++) {
        if (A.tile_size(TransA ? 0 : 1)[i] != B.block_dim(i)) {
            throw EINSUMSEXCEPTION("Tile sizes need to match between all three tensors.");
        }
    }
    int x_size = C->grid_size(0), y_size = C->grid_size(1);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            if constexpr (TransA) {
                if (C->has_zero_size(i, j) || !A.has_tile(j, i) || A.has_zero_size(j, i) || B.block_dim(j) == 0) {
                    continue;
                }
                C->lock();
                auto &C_tile = C->tile(i, j);
                C->unlock();
                gemm<TransA, TransB>(alpha, A.tile(j, i), B[j], beta, &C_tile);
            } else {
                if (C->has_zero_size(i, j) || !A.has_tile(i, j) || A.has_zero_size(i, j) || B.block_dim(j) == 0) {
                    continue;
                }
                C->lock();
                auto &C_tile = C->tile(i, j);
                C->unlock();
                gemm<TransA, TransB>(alpha, A.tile(j, i), B[j], beta, &C_tile);
            }
        }
    }
}

} // namespace einsums::linear_algebra::detail