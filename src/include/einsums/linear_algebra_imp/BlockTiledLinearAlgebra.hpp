#pragma once

#include "einsums/BlockTensor.hpp"
#include "einsums/TiledTensor.hpp"
#include "einsums/linear_algebra_imp/BaseLinearAlgebra.hpp"
#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

namespace einsums::linear_algebra::detail {

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires(RankTiledTensor<AType<T, Rank>, Rank, T> || RankTiledTensor<BType<T, Rank>, Rank, T> ||
                 RankTiledTensor<CType<T, Rank>, Rank, T>);
        requires(RankBlockTensor<AType<T, Rank>, Rank, T> || RankBlockTensor<BType<T, Rank>, Rank, T> ||
                 RankBlockTensor<CType<T, Rank>, Rank, T>);
        requires !(RankTiledTensor<AType<T, Rank>, Rank, T> && RankTiledTensor<BType<T, Rank>, Rank, T> &&
                   RankBlockTensor<CType<T, Rank>, Rank, T>);
    }
void direct_product(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    size_t num_blocks;
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T>) {
        num_blocks = A.num_blocks();
    } else {
        num_blocks = B.num_blocks();
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < num_blocks; i++) {
        std::array<int, Rank> index = std::array<int, Rank>{i};

        if constexpr (einsums::detail::IsIncoreRankTiledTensorV<AType<T, Rank>, Rank, T> &&
                      einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                      einsums::detail::IsIncoreRankTiledTensorV<CType<T, Rank>, Rank, T>) {
            if (!A.has_tile(index) || B.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.tile(index), B.block(i), beta, &(C->tile(index)));
        } else if constexpr (einsums::detail::IsIncoreRankTiledTensorV<AType<T, Rank>, Rank, T> &&
                             einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                             einsums::detail::IsIncoreRankBlockTensorV<CType<T, Rank>, Rank, T>) {
            if (!A.has_tile(index) || B.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.tile(index), B.block(i), beta, &(C->block(i)));
        } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                             einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                             einsums::detail::IsIncoreRankTiledTensorV<CType<T, Rank>, Rank, T>) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.block(i), B.block(i), beta, &(C->tile(index)));
        } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                             einsums::detail::IsIncoreRankTiledTensorV<BType<T, Rank>, Rank, T> &&
                             einsums::detail::IsIncoreRankTiledTensorV<CType<T, Rank>, Rank, T>) {
            if (!B.has_tile(index) || A.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.block(i), B.tile(index), beta, &(C->tile(index)));
        } else {
            if (!B.has_tile(index) || A.block_dim(i) == 0) {
                continue;
            }
            direct_product(alpha, A.block(i), B.tile(index), beta, &(C->block(i)));
        }
    }
}
} // namespace einsums::linear_algebra::detail