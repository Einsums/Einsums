#pragma once

#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/DeviceTensor.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace einsums::tensor_algebra {
template <TensorConcept AType, TensorConcept BType, typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires AType::Rank == sizeof...(AIndices);
        requires BType::Rank == sizeof...(BIndices);
    }
auto khatri_rao(const std::tuple<AIndices...> &, const AType &A, const std::tuple<BIndices...> &,
                const BType &B) -> BasicTensorLike<AType, typename AType::data_type, 2> {
    using OutType = BasicTensorLike<AType, typename AType::data_type, 2>;
    using T       = typename AType::data_type;
    LabeledSection0();

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();

    // Determine the common indices between A and B
    constexpr auto common = intersect_t<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    // Determine unique indices in A
    constexpr auto A_only = difference_t<std::tuple<AIndices...>, decltype(common)>();
    // Determine unique indices in B
    constexpr auto B_only = difference_t<std::tuple<BIndices...>, decltype(common)>();

    // Record the positions of each types.
    constexpr auto A_common_position = detail::find_type_with_position(common, A_indices);
    constexpr auto B_common_position = detail::find_type_with_position(common, B_indices);
    constexpr auto A_only_position   = detail::find_type_with_position(A_only, A_indices);
    constexpr auto B_only_position   = detail::find_type_with_position(B_only, B_indices);

    // Obtain dimensions of the indices discovered above
    auto A_common_dims = detail::get_dim_for(A, A_common_position);
    auto B_common_dims = detail::get_dim_for(B, B_common_position);
    auto A_only_dims   = detail::get_dim_for(A, A_only_position);
    auto B_only_dims   = detail::get_dim_for(B, B_only_position);

    // Sanity check - ensure the common dims between A and B are the same size.
    for_sequence<std::tuple_size_v<decltype(common)>>([&](auto i) {
        if (std::get<i>(A_common_dims) != std::get<i>(B_common_dims)) {
            throw EINSUMSEXCEPTION(fmt::format("Common dimensions for index {} of A and B do not match.", std::get<i>(common)));
        }
    });

#ifdef __HIP__
    if constexpr (std::is_same_v<OutType, DeviceTensor<T, 2>>) {
        auto result_dims = std::tuple_cat(std::make_tuple("KR product"), std::make_tuple(einsums::detail::DEV_ONLY), A_only_dims,
                                          B_only_dims, A_common_dims);
        // Construct resulting tensor
        auto result = std::make_from_tuple<DeviceTensor<T, std::tuple_size_v<decltype(result_dims)> - 2>>(result_dims);
        // Perform the actual Khatri-Rao product using our einsum routine.
        einsum(std::tuple_cat(A_only, B_only, common), &result, std::tuple_cat(A_only, common), A, std::tuple_cat(B_only, common), B);

        // Return a reconstruction of the result tensor ... this can be considered as a simple reshape of the tensor.

        return OutType{std::move(result), "KR product", -1, detail::product_dims(A_common_position, A)};
    } else {
#endif
        auto result_dims = std::tuple_cat(std::make_tuple("KR product"), A_only_dims, B_only_dims, A_common_dims);
        // Construct resulting tensor
        auto result = std::make_from_tuple<Tensor<T, std::tuple_size_v<decltype(result_dims)> - 1>>(result_dims);
        // Perform the actual Khatri-Rao product using our einsum routine.
        einsum(std::tuple_cat(A_only, B_only, common), &result, std::tuple_cat(A_only, common), A, std::tuple_cat(B_only, common), B);

        // Return a reconstruction of the result tensor ... this can be considered as a simple reshape of the tensor.

        return OutType{std::move(result), "KR product", -1, detail::product_dims(A_common_position, A)};
#ifdef __HIP__
    }
#endif
}
} // namespace einsums::tensor_algebra