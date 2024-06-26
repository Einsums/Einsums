#pragma once

namespace einsums::tensor_algebra::detail {
template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
    requires requires {
        requires RankBlockTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires RankBlockTensor<BType<BDataType, BRank>, BRank, BDataType>;
        requires RankBlockTensor<CType<CDataType, CRank>, CRank, CDataType>;
    }
auto einsum_special_dispatch(const CDataType C_prefactor, const std::tuple<CIndices...> &C_indices, CType<CDataType, CRank> *C,
                             const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
                             const std::tuple<AIndices...> &A_indices, const AType<ADataType, ARank> &A,
                             const std::tuple<BIndices...> &B_indices, const BType<BDataType, BRank> &B) -> void {
}
} // namespace einsums::tensor_algebra::detail