//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"
#include "einsums/_TensorAlgebraUtilities.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/DeviceSort.hpp"
#endif

namespace einsums::tensor_algebra {

#if defined(EINSUMS_USE_HPTT)

namespace detail {

void EINSUMS_EXPORT sort(const int *perm, const int dim, const float alpha, const float *A, const int *sizeA, const float beta, float *B);
void EINSUMS_EXPORT sort(const int *perm, const int dim, const double alpha, const double *A, const int *sizeA, const double beta,
                         double *B);
void EINSUMS_EXPORT sort(const int *perm, const int dim, const std::complex<float> alpha, const std::complex<float> *A, const int *sizeA,
                         const std::complex<float> beta, std::complex<float> *B);
void EINSUMS_EXPORT sort(const int *perm, const int dim, const std::complex<double> alpha, const std::complex<double> *A, const int *sizeA,
                         const std::complex<double> beta, std::complex<double> *B);

} // namespace detail

#endif

//
// sort algorithm
//
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename CType, size_t CRank,
          typename... CIndices, typename... AIndices, typename U, typename T = double>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, ARank, T>;
        requires CoreRankTensor<CType<T, CRank>, CRank, T>;
    }
auto sort(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<T, CRank> *C, const U UA_prefactor,
          const std::tuple<AIndices...> &A_indices,
          const AType<T, ARank> &A) -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, CRank>, CType<T, CRank>> &&
                                                        std::is_base_of_v<::einsums::detail::TensorBase<T, ARank>, AType<T, ARank>> &&
                                                        sizeof...(CIndices) == sizeof...(AIndices) && sizeof...(CIndices) == CRank &&
                                                        sizeof...(AIndices) == ARank && std::is_arithmetic_v<U>> {

    LabeledSection1((std::fabs(UC_prefactor) > EINSUMS_ZERO)
                        ? fmt::format(R"(sort: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor,
                                      A.name(), print_tuple_no_type(A_indices), UC_prefactor, C->name(), print_tuple_no_type(C_indices))
                        : fmt::format(R"(sort: "{}"{} = {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor, A.name(),
                                      print_tuple_no_type(A_indices)));

    const T C_prefactor = UC_prefactor;
    const T A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a sort
    constexpr auto check = difference_t<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = detail::find_type_with_position(C_indices, A_indices);

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto a_dims      = detail::get_dim_ranges_for(A, target_position_in_A);

    // If the prefactor is zero, set the tensor to zero. This avoids NaNs.
    if (C_prefactor == T(0.0)) {
        *C = T(0.0);
    }

    // HPTT interface currently only works for full Tensors and not TensorViews
#if defined(EINSUMS_USE_HPTT)
    if constexpr (std::is_same_v<CType<T, CRank>, Tensor<T, CRank>> && std::is_same_v<AType<T, ARank>, Tensor<T, ARank>>) {
        std::array<int, ARank> perms{};
        std::array<int, ARank> size{};

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0] = get_from_tuple<unsigned long>(target_position_in_A, (2 * i0) + 1);
            size[i0]  = A.dim(i0);
        }

        detail::sort(perms.data(), ARank, A_prefactor, A.data(), size.data(), C_prefactor, C->data());
    } else
#endif
        if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)>) {
        linear_algebra::axpby(A_prefactor, A, C_prefactor, C);
    } else {
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

        EINSUMS_OMP_PARALLEL_FOR
        for (auto it = view.begin(); it < view.end(); it++) {
            auto A_order = detail::construct_indices<AIndices...>(*it, target_position_in_A, *it, target_position_in_A);

            T &target_value = std::apply(*C, *it);
            T  A_value      = std::apply(A, A_order);

            target_value = C_prefactor * target_value + A_prefactor * A_value;
        }
    }
}

// Sort with default values, no smart pointers
template <NotASmartPointer ObjectA, NotASmartPointer ObjectC, typename... CIndices, typename... AIndices>
void sort(const std::tuple<CIndices...> &C_indices, ObjectC *C, const std::tuple<AIndices...> &A_indices, const ObjectA &A) {
    sort(0, C_indices, C, 1, A_indices, A);
}

// Sort with default values, two smart pointers
template <SmartPointer SmartPointerA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void sort(const std::tuple<CIndices...> &C_indices, SmartPointerC *C, const std::tuple<AIndices...> &A_indices, const SmartPointerA &A) {
    sort(0, C_indices, C->get(), 1, A_indices, *A);
}

// Sort with default values, one smart pointer (A)
template <SmartPointer SmartPointerA, NotASmartPointer PointerC, typename... CIndices, typename... AIndices>
void sort(const std::tuple<CIndices...> &C_indices, PointerC *C, const std::tuple<AIndices...> &A_indices, const SmartPointerA &A) {
    sort(0, C_indices, C, 1, A_indices, *A);
}

// Sort with default values, one smart pointer (C)
template <NotASmartPointer ObjectA, SmartPointer SmartPointerC, typename... CIndices, typename... AIndices>
void sort(const std::tuple<CIndices...> &C_indices, SmartPointerC *C, const std::tuple<AIndices...> &A_indices, const ObjectA &A) {
    sort(0, C_indices, C->get(), 1, A_indices, A);
}

} // namespace einsums::tensor_algebra