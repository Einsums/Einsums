#pragma once

#include "einsums/TensorAlgebra.hpp"
#include "einsums/tensor_algebra_backends/Dispatch.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <tuple>

namespace einsums::tensor_algebra::detail {

template <bool OnlyUseGenericAlgorithm, TensorConcept AType, TensorConcept BType, TensorConcept CType, typename... CIndices,
          typename... AIndices, typename... BIndices>
    requires requires {
        requires(TiledTensorConcept<AType> || BlockTensorConcept<AType>);
        requires(TiledTensorConcept<BType> || BlockTensorConcept<BType>);
        requires(TiledTensorConcept<CType> || BlockTensorConcept<CType>);
        requires(!TiledTensorConcept<AType> || !TiledTensorConcept<BType> || !TiledTensorConcept<CType>);
        requires(!BlockTensorConcept<AType> || !BlockTensorConcept<BType> || !BlockTensorConcept<CType>);
    }
auto einsum_special_dispatch(const typename CType::data_type C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C,
                             const BiggestTypeT<typename AType::data_type, typename BType::data_type> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices,
                             const BType &B) -> void {

    using ADataType        = typename AType::data_type;
    using BDataType        = typename BType::data_type;
    using CDataType        = typename CType::data_type;
    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = CType::Rank;

    constexpr auto unique_indices = unique_t<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides);

    std::array<int, ARank> A_index_table;
    std::array<int, BRank> B_index_table;
    std::array<int, CRank> C_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    if (C_prefactor == CDataType(0.0)) {
        if constexpr (einsums::detail::IsTiledTensorV<CType> && einsums::detail::IsTensorViewV<CType>) {
            C->zero_no_clear();
        } else {
            C->zero();
        }
    } else {
        *C *= C_prefactor;
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        thread_local std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        thread_local std::array<int, ARank> A_tile_index;
        thread_local std::array<int, BRank> B_tile_index;
        thread_local std::array<int, CRank> C_tile_index;

        bool skip = false;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
            if constexpr (einsums::detail::IsBlockTensorV<AType>) {
                if (A_tile_index[i] != A_tile_index[0]) {
                    skip = true;
                }
            }
        }

        if (skip) {
            continue;
        }

        for (int i = 0; i < BRank; i++) {
            B_tile_index[i] = unique_index_table[B_index_table[i]];
            if constexpr (einsums::detail::IsBlockTensorV<BType>) {
                if (B_tile_index[i] != B_tile_index[0]) {
                    skip = true;
                }
            }
        }

        if (skip) {
            continue;
        }

        for (int i = 0; i < CRank; i++) {
            C_tile_index[i] = unique_index_table[C_index_table[i]];
            if constexpr (einsums::detail::IsBlockTensorV<CType>) {
                if (C_tile_index[i] != C_tile_index[0]) {
                    skip = true;
                }
            }
        }

        if (skip) {
            continue;
        }

        if constexpr (einsums::detail::IsBlockTensorV<AType>) {
            if (A.block_dim(A_tile_index[0]) == 0) {
                continue;
            }
        } else {
            if (!A.has_tile(A_tile_index) || A.has_zero_size(A_tile_index)) {
                continue;
            }
        }

        if constexpr (einsums::detail::IsBlockTensorV<BType>) {
            if (B.block_dim(B_tile_index[0]) == 0) {
                continue;
            }
        } else {
            if (!B.has_tile(B_tile_index) || B.has_zero_size(B_tile_index)) {
                continue;
            }
        }

        if constexpr (einsums::detail::IsBlockTensorV<CType>) {
            if (C->block_dim(C_tile_index[0]) == 0) {
                continue;
            }

            C->lock();
            auto &C_block = C->block(C_tile_index[0]);
            C->unlock();
            C_block.lock();
            if constexpr (einsums::detail::IsBlockTensorV<AType>) {
                einsum<OnlyUseGenericAlgorithm>(CDataType{1.0}, C_indices, &C_block, AB_prefactor, A_indices, A[A_tile_index[0]], B_indices,
                                                B.tile(B_tile_index));
            } else if constexpr (einsums::detail::IsBlockTensorV<BType>) {
                einsum<OnlyUseGenericAlgorithm>(CDataType{1.0}, C_indices, &C_block, AB_prefactor, A_indices, A.tile(A_tile_index),
                                                B_indices, B[B_tile_index[0]]);
            }
            C_block.unlock();
        } else {
            if (C->has_zero_size(C_tile_index)) {
                continue;
            }

            C->lock();
            auto &C_tile = C->tile(C_tile_index);
            C->unlock();
            C_tile.lock();
            if constexpr (einsums::detail::IsBlockTensorV<AType> && einsums::detail::IsBlockTensorV<BType>) {
                einsum<OnlyUseGenericAlgorithm>(CDataType{1.0}, C_indices, &C_tile, AB_prefactor, A_indices, A[A_tile_index[0]], B_indices,
                                                B[B_tile_index[0]]);
            } else if constexpr (einsums::detail::IsBlockTensorV<AType>) {
                einsum<OnlyUseGenericAlgorithm>(CDataType{1.0}, C_indices, &C_tile, AB_prefactor, A_indices, A[A_tile_index[0]], B_indices,
                                                B.tile(B_tile_index));
            } else if constexpr (einsums::detail::IsBlockTensorV<BType>) {
                einsum<OnlyUseGenericAlgorithm>(CDataType{1.0}, C_indices, &C_tile, AB_prefactor, A_indices, A.tile(A_tile_index),
                                                B_indices, B[B_tile_index[0]]);
            }
            C_tile.unlock();
        }
    }
}

} // namespace einsums::tensor_algebra::detail