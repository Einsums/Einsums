#pragma once

#include "LinearAlgebra.hpp"
#include "OpenMP.h"
#include "Print.hpp"
#include "STL.hpp"
#include "Tensor.hpp"
#if defined(EINSUMS_USE_HPTT)
#include "hptt.h"
#endif
#include "range/v3/view/cartesian_product.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#undef I

namespace einsums::TensorAlgebra {

namespace Index {

#define MAKE_INDEX(x)                                                                                                                      \
    struct x {                                                                                                                             \
        static constexpr char letter = static_cast<const char (&)[2]>(#x)[0];                                                              \
        constexpr x() = default;                                                                                                           \
    };                                                                                                                                     \
    static struct x x;                                                                                                                     \
    inline auto operator<<(std::ostream &os, const struct x &)->std::ostream & {                                                           \
        os << x::letter;                                                                                                                   \
        return os;                                                                                                                         \
    }

MAKE_INDEX(A); // NOLINT
MAKE_INDEX(a); // NOLINT
MAKE_INDEX(B); // NOLINT
MAKE_INDEX(b); // NOLINT
MAKE_INDEX(C); // NOLINT
MAKE_INDEX(c); // NOLINT
MAKE_INDEX(D); // NOLINT
MAKE_INDEX(d); // NOLINT
MAKE_INDEX(E); // NOLINT
MAKE_INDEX(e); // NOLINT
MAKE_INDEX(F); // NOLINT
MAKE_INDEX(f); // NOLINT

MAKE_INDEX(I); // NOLINT
MAKE_INDEX(i); // NOLINT
MAKE_INDEX(J); // NOLINT
MAKE_INDEX(j); // NOLINT
MAKE_INDEX(K); // NOLINT
MAKE_INDEX(k); // NOLINT
MAKE_INDEX(L); // NOLINT
MAKE_INDEX(l); // NOLINT
MAKE_INDEX(M); // NOLINT
MAKE_INDEX(m); // NOLINT
MAKE_INDEX(N); // NOLINT
MAKE_INDEX(n); // NOLINT

MAKE_INDEX(P); // NOLINT
MAKE_INDEX(p); // NOLINT
MAKE_INDEX(Q); // NOLINT
MAKE_INDEX(q); // NOLINT
MAKE_INDEX(R); // NOLINT
MAKE_INDEX(r); // NOLINT
MAKE_INDEX(S); // NOLINT
MAKE_INDEX(s); // NOLINT

#undef MAKE_INDEX
} // namespace Index

template <typename... Args>
struct Indices : public std::tuple<Args...> {
    Indices(Args... args) : std::tuple<Args...>(args...){};
};

namespace Detail {

template <size_t Rank, typename... Args, std::size_t... I>
auto order_indices(const std::tuple<Args...> &combination, const std::array<size_t, Rank> &order, std::index_sequence<I...>) {
    return std::tuple{get_from_tuple<size_t>(combination, order[I])...};
}

} // namespace Detail

template <size_t Rank, typename... Args>
auto order_indices(const std::tuple<Args...> &combination, const std::array<size_t, Rank> &order) {
    return Detail::order_indices(combination, order, std::make_index_sequence<Rank>{});
}

namespace Detail {

template <typename T, int Position>
constexpr auto _find_type_with_position() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto _find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position), _find_type_with_position<T, Position + 1, Args...>());
    } else {
        return _find_type_with_position<T, Position + 1, Args...>();
    }
}

template <template <size_t, typename> typename TensorType, size_t Rank, typename... Args, std::size_t... I, typename T = double>
auto get_dim_ranges_for(const TensorType<Rank, T> &tensor, const std::tuple<Args...> &args, std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(std::get<2 * I + 1>(args)))...};
}

template <typename T, int Position>
constexpr auto find_position() {
    return -1;
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto find_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        // Found it
        return Position;
    } else {
        return find_position<T, Position + 1, Args...>();
    }
}

template <typename AIndex, typename... Args>
constexpr auto find_position() {
    return find_position<AIndex, 0, Args...>();
}

template <typename AIndex, typename... TargetCombination>
constexpr auto find_position(const std::tuple<TargetCombination...> &) {
    return Detail::find_position<AIndex, TargetCombination...>();
}

template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(Detail::_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

template <typename... Ts, typename... Us>
constexpr auto find_type_with_position(const std::tuple<Ts...> &, const std::tuple<Us...> &) {
    return _find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

template <template <size_t, typename> typename TensorType, size_t Rank, typename... Args, typename T = double>
auto get_dim_ranges_for(const TensorType<Rank, T> &tensor, const std::tuple<Args...> &args) {
    return Detail::get_dim_ranges_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

template <typename AIndex, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
auto construct_index(const std::tuple<TargetCombination...> &target_combination, const std::tuple<TargetPositionInC...> &,
                     const std::tuple<LinkCombination...> &link_combination, const std::tuple<LinkPositionInLink...> &) {

    constexpr auto IsAIndexInC = Detail::find_position<AIndex, TargetPositionInC...>();
    constexpr auto IsAIndexInLink = Detail::find_position<AIndex, LinkPositionInLink...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1, "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC / 2>(target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink / 2>(link_combination);
    } else {
        return -1;
    }
}

template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto
construct_indices(const std::tuple<TargetCombination...> &target_combination, const std::tuple<TargetPositionInC...> &target_position_in_C,
                  const std::tuple<LinkCombination...> &link_combination, const std::tuple<LinkPositionInLink...> &link_position_in_link) {
    return std::make_tuple(construct_index<AIndices>(target_combination, target_position_in_C, link_combination, link_position_in_link)...);
}

template <typename... PositionsInX, std::size_t... I>
constexpr auto _contiguous_positions(const std::tuple<PositionsInX...> &x, std::index_sequence<I...>) -> bool {
    return ((std::get<2 * I + 1>(x) == std::get<2 * I + 3>(x) - 1) && ... && true);
}

template <typename... PositionsInX>
constexpr auto contiguous_positions(const std::tuple<PositionsInX...> &x) -> bool {
    if constexpr (sizeof...(PositionsInX) <= 2) {
        return true;
    } else {
        return _contiguous_positions(x, std::make_index_sequence<sizeof...(PositionsInX) / 2 - 1>{});
    }
}

template <typename... PositionsInX, typename... PositionsInY, std::size_t... I>
constexpr auto _is_same_ordering(const std::tuple<PositionsInX...> &positions_in_x, const std::tuple<PositionsInY...> &positions_in_y,
                                 std::index_sequence<I...>) {
    return (std::is_same_v<decltype(std::get<2 * I>(positions_in_x)), decltype(std::get<2 * I>(positions_in_y))> && ...);
}

template <typename... PositionsInX, typename... PositionsInY>
constexpr auto is_same_ordering(const std::tuple<PositionsInX...> &positions_in_x, const std::tuple<PositionsInY...> &positions_in_y) {
    // static_assert(sizeof...(PositionsInX) == sizeof...(PositionsInY) && sizeof...(PositionsInX) > 0);
    if constexpr (sizeof...(PositionsInX) == 0 || sizeof...(PositionsInY) == 0)
        return false; // NOLINT
    else if constexpr (sizeof...(PositionsInX) != sizeof...(PositionsInY))
        return false;
    else
        return _is_same_ordering(positions_in_x, positions_in_y, std::make_index_sequence<sizeof...(PositionsInX) / 2>{});
}

template <template <size_t, typename> typename XType, size_t XRank, typename... PositionsInX, std::size_t... I, typename T = double>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType<XRank, T> &X, std::index_sequence<I...>) -> size_t {
    return (X.dim(std::get<2 * I + 1>(indices)) * ... * 1);
}

template <template <size_t, typename> typename XType, size_t XRank, typename... PositionsInX, std::size_t... I, typename T = double>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType<XRank, T> &X, std::index_sequence<I...>) -> bool {
    return ((X.dim(std::get<1>(indices)) == X.dim(std::get<2 * I + 1>(indices))) && ... && 1);
}

template <typename LHS, typename RHS, std::size_t... I>
constexpr auto same_indices(std::index_sequence<I...>) {
    return (std::is_same_v<std::tuple_element_t<I, LHS>, std::tuple_element_t<I, RHS>> && ...);
}

template <template <size_t, typename> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto product_dims(const std::tuple<PositionsInX...> &indices, const XType<XRank, T> &X) -> size_t {
    return Detail::product_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template <template <size_t, typename> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto is_same_dims(const std::tuple<PositionsInX...> &indices, const XType<XRank, T> &X) -> size_t {
    return Detail::is_same_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template <template <size_t, typename> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto last_stride(const std::tuple<PositionsInX...> &indices, const XType<XRank, T> &X) -> size_t {
    return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
}

template <typename LHS, typename RHS>
constexpr auto same_indices() {
    if constexpr (std::tuple_size_v<LHS> != std::tuple_size_v<RHS>)
        return false;
    else
        return Detail::same_indices<LHS, RHS>(std::make_index_sequence<std::tuple_size_v<LHS>>());
}

template <typename T, typename... CIndices, typename... AIndices, typename... BIndices, typename... TargetDims, typename... LinkDims,
          typename... TargetPositionInC, typename... LinkPositionInLink, template <size_t, typename> typename CType, size_t CRank,
          template <size_t, typename> typename AType, size_t ARank, template <size_t, typename> typename BType, size_t BRank>
void einsum_generic_algorithm(const std::tuple<CIndices...> & /*C_indices*/, const std::tuple<AIndices...> & /*A_indices*/,
                              const std::tuple<BIndices...> & /*B_indices*/, const std::tuple<TargetDims...> &target_dims,
                              const std::tuple<LinkDims...> &link_dims, const std::tuple<TargetPositionInC...> &target_position_in_C,
                              const std::tuple<LinkPositionInLink...> &link_position_in_link, const T C_prefactor, CType<CRank, T> *C,
                              const T AB_prefactor, const AType<ARank, T> &A, const BType<BRank, T> &B) {
    Timer::push("generic algorithm");

    auto view = std::apply(ranges::views::cartesian_product, target_dims);

    if constexpr (sizeof...(LinkDims) != 0) {
#pragma omp parallel for simd
        for (auto it = view.begin(); it < view.end(); it++) {
            // println("target_combination: {}", print_tuple_no_type(target_combination));
            auto C_order = Detail::construct_indices<CIndices...>(*it, target_position_in_C, std::tuple<>(), target_position_in_C);
            // println("C_order: {}", print_tuple_no_type(C_order));

            // This is the generic case.
            T sum{0};
            for (auto link_combination : std::apply(ranges::views::cartesian_product, link_dims)) {
                // Print::Indent _indent;

                // Construct the tuples that will be used to access the tensor elements of A and B
                auto A_order = Detail::construct_indices<AIndices...>(*it, target_position_in_C, link_combination, link_position_in_link);
                auto B_order = Detail::construct_indices<BIndices...>(*it, target_position_in_C, link_combination, link_position_in_link);

                // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
                T A_value = std::apply(A, A_order);
                T B_value = std::apply(B, B_order);

                sum += AB_prefactor * A_value * B_value;
            }

            T &target_value = std::apply(*C, C_order);
            target_value *= C_prefactor;
            target_value += sum;
        }
    } else {
#pragma omp parallel for simd
        for (auto it = view.begin(); it < view.end(); it++) {

            // This is the generic case.
            T sum{0};

            // Construct the tuples that will be used to access the tensor elements of A and B
            auto A_order = Detail::construct_indices<AIndices...>(*it, target_position_in_C, std::tuple<>(), target_position_in_C);
            auto B_order = Detail::construct_indices<BIndices...>(*it, target_position_in_C, std::tuple<>(), target_position_in_C);
            auto C_order = Detail::construct_indices<CIndices...>(*it, target_position_in_C, std::tuple<>(), target_position_in_C);

            // Get the tensor element using the operator()(MultiIndex...) function of Tensor.
            T A_value = std::apply(A, A_order);
            T B_value = std::apply(B, B_order);

            sum += AB_prefactor * A_value * B_value;

            T &target_value = std::apply(*C, C_order);
            target_value *= C_prefactor;
            target_value += sum;
        }
    }
    Timer::pop();
}

template <bool OnlyUseGenericAlgorithm, template <size_t, typename> typename AType, size_t ARank,
          template <size_t, typename> typename BType, size_t BRank, template <size_t, typename> typename CType, size_t CRank,
          typename... CIndices, typename... AIndices, typename... BIndices, typename T = double>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> & /*Cs*/, CType<CRank, T> *C, const T AB_prefactor,
            const std::tuple<AIndices...> & /*As*/, const AType<ARank, T> &A, const std::tuple<BIndices...> & /*Bs*/,
            const BType<BRank, T> &B) -> std::enable_if_t<std::is_base_of_v<::einsums::Detail::TensorBase<ARank, T>, AType<ARank, T>> &&
                                                          std::is_base_of_v<::einsums::Detail::TensorBase<BRank, T>, BType<BRank, T>> &&
                                                          std::is_base_of_v<::einsums::Detail::TensorBase<CRank, T>, CType<CRank, T>>> {
    Print::Indent _indent;

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();
    constexpr auto C_indices = std::tuple<CIndices...>();

    // 1. Ensure the ranks are correct. (Compile-time check.)
    static_assert(sizeof...(CIndices) == CRank, "Rank of C does not match Indices given for C.");
    static_assert(sizeof...(AIndices) == ARank, "Rank of A does not match Indices given for A.");
    static_assert(sizeof...(BIndices) == BRank, "Rank of B does not match Indices given for B.");

    // 2. Determine the links from AIndices and BIndices
    constexpr auto links0 = intersect_t<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    // 2a. Remove any links that appear in the target
    constexpr auto links = difference_t<decltype(links0), std::tuple<CIndices...>>();

    // 3. Determine the links between CIndices and AIndices
    constexpr auto CAlinks = intersect_t<std::tuple<CIndices...>, std::tuple<AIndices...>>();

    // 4. Determine the links between CIndices and BIndices
    constexpr auto CBlinks = intersect_t<std::tuple<CIndices...>, std::tuple<BIndices...>>();

    // Determine unique indices in A
    constexpr auto A_only = difference_t<std::tuple<AIndices...>, decltype(links)>();
    constexpr auto B_only = difference_t<std::tuple<BIndices...>, decltype(links)>();

    constexpr auto A_unique = unique_t<std::tuple<AIndices...>>();
    constexpr auto B_unique = unique_t<std::tuple<BIndices...>>();
    constexpr auto C_unique = unique_t<std::tuple<CIndices...>>();
    constexpr auto link_unique = c_unique_t<decltype(links)>();

    constexpr bool A_hadamard_found = std::tuple_size_v<std::tuple<AIndices...>> != std::tuple_size_v<decltype(A_unique)>;
    constexpr bool B_hadamard_found = std::tuple_size_v<std::tuple<BIndices...>> != std::tuple_size_v<decltype(B_unique)>;
    constexpr bool C_hadamard_found = std::tuple_size_v<std::tuple<CIndices...>> != std::tuple_size_v<decltype(C_unique)>;

    constexpr auto link_position_in_A = Detail::find_type_with_position(links, A_indices);
    constexpr auto link_position_in_B = Detail::find_type_with_position(links, B_indices);
    constexpr auto link_position_in_link = Detail::find_type_with_position(links, links);

    constexpr auto target_position_in_A = Detail::find_type_with_position(C_indices, A_indices);
    constexpr auto target_position_in_B = Detail::find_type_with_position(C_indices, B_indices);
    constexpr auto target_position_in_C = Detail::find_type_with_position(C_indices, C_indices);

    constexpr auto A_target_position_in_C = Detail::find_type_with_position(A_indices, C_indices);
    constexpr auto B_target_position_in_C = Detail::find_type_with_position(B_indices, C_indices);

    auto target_dims = Detail::get_dim_ranges_for(*C, Detail::find_type_with_position(C_unique, C_indices));
    auto link_dims = Detail::get_dim_ranges_for(A, link_position_in_A);

    constexpr auto contiguous_link_position_in_A = Detail::contiguous_positions(link_position_in_A);
    constexpr auto contiguous_link_position_in_B = Detail::contiguous_positions(link_position_in_B);

    constexpr auto contiguous_target_position_in_A = Detail::contiguous_positions(target_position_in_A);
    constexpr auto contiguous_target_position_in_B = Detail::contiguous_positions(target_position_in_B);

    constexpr auto contiguous_A_targets_in_C = Detail::contiguous_positions(A_target_position_in_C);
    constexpr auto contiguous_B_targets_in_C = Detail::contiguous_positions(B_target_position_in_C);

    constexpr auto same_ordering_link_position_in_AB = Detail::is_same_ordering(link_position_in_A, link_position_in_B);
    constexpr auto same_ordering_target_position_in_CA = Detail::is_same_ordering(target_position_in_A, A_target_position_in_C);
    constexpr auto same_ordering_target_position_in_CB = Detail::is_same_ordering(target_position_in_B, B_target_position_in_C);

    constexpr auto C_exactly_matches_A =
        sizeof...(CIndices) == sizeof...(AIndices) && same_indices<std::tuple<CIndices...>, std::tuple<AIndices...>>();
    constexpr auto C_exactly_matches_B =
        sizeof...(CIndices) == sizeof...(BIndices) && same_indices<std::tuple<CIndices...>, std::tuple<BIndices...>>();
    constexpr auto A_exactly_matches_B = same_indices<std::tuple<AIndices...>, std::tuple<BIndices...>>();

    constexpr auto is_gemm_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                                      contiguous_target_position_in_B && contiguous_A_targets_in_C && contiguous_B_targets_in_C &&
                                      same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                                      same_ordering_target_position_in_CB && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    constexpr auto is_gemv_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                                      same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                                      !same_ordering_target_position_in_CB && std::tuple_size_v<decltype(B_target_position_in_C)> == 0 &&
                                      !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    constexpr auto element_wise_multiplication =
        C_exactly_matches_A && C_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    constexpr auto dot_product =
        sizeof...(CIndices) == 0 && A_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    constexpr auto outer_product = std::tuple_size_v<decltype(links)> == 0 && contiguous_target_position_in_A &&
                                   contiguous_target_position_in_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;

    if constexpr (dot_product) {
        T temp = LinearAlgebra::dot(A, B);
        (*C) *= C_prefactor;
        (*C) += AB_prefactor * temp;

        return;
    } else if constexpr (element_wise_multiplication) {
        Dim<1> common;
        common[0] = 1;
        for (auto el : C->dims()) {
            common[0] *= el;
        }

        TensorView<1, T> tC{*C, common};
        const TensorView<1, T> tA{const_cast<AType<ARank, T> &>(A), common}, tB{const_cast<BType<BRank, T> &>(B), common};

        for (size_t i = 0; i < common[0]; i++) {
            T temp = AB_prefactor * tA(i) * tB(i);
            T &target_value = tC(i);
            target_value *= C_prefactor;
            target_value += temp;
        }

        return;
    } else if constexpr (outer_product) {
        do { // do {} while (false) trick to allow us to use a break below to "break" out of the loop.
            constexpr bool swap_AB = std::get<1>(A_target_position_in_C) != 0;

            Dim<2> dC;
            dC[0] = product_dims(A_target_position_in_C, *C);
            dC[1] = product_dims(B_target_position_in_C, *C);
            if constexpr (swap_AB)
                std::swap(dC[0], dC[1]);

            TensorView<2, T> tC{*C, dC};

            if (C_prefactor != T{1.0})
                LinearAlgebra::scale(C_prefactor, C);

            try {
                if constexpr (swap_AB) {
                    LinearAlgebra::ger(AB_prefactor, B.to_rank_1_view(), A.to_rank_1_view(), &tC);
                } else {
                    LinearAlgebra::ger(AB_prefactor, A.to_rank_1_view(), B.to_rank_1_view(), &tC);
                }
            } catch (std::runtime_error &e) {
                // TODO: If ger throws exception the timer gets out of sync.
                Timer::pop();
#if defined(EINSUMS_SHOW_WARNING)
                println(
                    bg(fmt::color::yellow) | fg(fmt::color::black),
                    "Optimized outer product failed. Likely from a non-contiguous TensorView. Attempting to perform generic algorithm.");
#endif
                if (C_prefactor == T{0.0}) {
#if defined(EINSUMS_SHOW_WARNING)
                    println(bg(fmt::color::red) | fg(fmt::color::white),
                            "WARNING!! Unable to undo C_prefactor ({}) on C ({}) tensor. Check your results!!!", C_prefactor, C->name());
#endif
                } else {
                    LinearAlgebra::scale(1.0 / C_prefactor, C);
                }
                break; // out of the do {} while(false) loop.
            }
            // If we got to this position, assume we successfully called ger.
            return;
        } while (false);
    } else if constexpr (!OnlyUseGenericAlgorithm) {
        do { // do {} while (false) trick to allow us to use a break below to "break" out of the loop.

            // To use a gemm the input tensors need to be at least rank 2
            if constexpr (CRank >= 2 && ARank >= 2 && BRank >= 2) {
                if constexpr (!A_hadamard_found && !B_hadamard_found && !C_hadamard_found) {
                    if constexpr (is_gemv_possible) {
                        constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;

                        Dim<2> dA;
                        Dim<1> dB, dC;

                        dA[0] = product_dims(A_target_position_in_C, *C);
                        dA[1] = product_dims(link_position_in_A, A);

                        dB[0] = product_dims(link_position_in_A, A);
                        dC[0] = product_dims(A_target_position_in_C, *C);

                        const TensorView<2, T> tA{const_cast<AType<ARank, T> &>(A), dA};
                        const TensorView<1, T> tB{const_cast<BType<BRank, T> &>(B), dB};
                        TensorView<1, T> tC{*C, dC};

                        if constexpr (transpose_A) {
                            LinearAlgebra::gemv<true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                        } else {
                            LinearAlgebra::gemv<false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                        }

                        return;
                    } else if constexpr (is_gemm_possible) {

                        if (!C->full_view_of_underlying() || !A.full_view_of_underlying() || !B.full_view_of_underlying()) {
                            // Fall through to generic algorithm.
                            break;
                        }

                        constexpr bool transpose_A = std::get<1>(link_position_in_A) == 0;
                        constexpr bool transpose_B = std::get<1>(link_position_in_B) != 0;
                        constexpr bool transpose_C = std::get<1>(A_target_position_in_C) != 0;

                        Dim<2> dA, dB, dC;
                        Stride<2> sA, sB, sC;

                        dA[0] = product_dims(A_target_position_in_C, *C);
                        dA[1] = product_dims(link_position_in_A, A);
                        sA[0] = last_stride(target_position_in_A, A);
                        sA[1] = last_stride(link_position_in_A, A);
                        if constexpr (transpose_A) {
                            std::swap(dA[0], dA[1]);
                            std::swap(sA[0], sA[1]);
                        }

                        dB[0] = product_dims(link_position_in_B, B);
                        dB[1] = product_dims(B_target_position_in_C, *C);
                        sB[0] = last_stride(link_position_in_B, B);
                        sB[1] = last_stride(target_position_in_B, B);
                        if constexpr (transpose_B) {
                            std::swap(dB[0], dB[1]);
                            std::swap(sB[0], sB[1]);
                        }

                        dC[0] = product_dims(A_target_position_in_C, *C);
                        dC[1] = product_dims(B_target_position_in_C, *C);
                        sC[0] = last_stride(A_target_position_in_C, *C);
                        sC[1] = last_stride(B_target_position_in_C, *C);
                        if constexpr (transpose_C) {
                            std::swap(dC[0], dC[1]);
                            std::swap(sC[0], sC[1]);
                        }

                        TensorView<2, T> tC{*C, dC, sC};
                        const TensorView<2, T> tA{const_cast<AType<ARank, T> &>(A), dA, sA}, tB{const_cast<BType<BRank, T> &>(B), dB, sB};

                        // println("--------------------");
                        // println(*C);
                        // println(tC);
                        // println("--------------------");
                        // println(A);
                        // println(tA);
                        // println("--------------------");
                        // println(B);
                        // println(tB);
                        // println("--------------------");

                        if constexpr (!transpose_C && !transpose_A && !transpose_B) {
                            LinearAlgebra::gemm<false, false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && !transpose_A && transpose_B) {
                            LinearAlgebra::gemm<false, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && transpose_A && !transpose_B) {
                            LinearAlgebra::gemm<true, false>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (!transpose_C && transpose_A && transpose_B) {
                            LinearAlgebra::gemm<true, true>(AB_prefactor, tA, tB, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && !transpose_A && !transpose_B) {
                            LinearAlgebra::gemm<true, true>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && !transpose_A && transpose_B) {
                            LinearAlgebra::gemm<false, true>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && transpose_A && !transpose_B) {
                            LinearAlgebra::gemm<true, false>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else if constexpr (transpose_C && transpose_A && transpose_B) {
                            LinearAlgebra::gemm<false, false>(AB_prefactor, tB, tA, C_prefactor, &tC);
                            return;
                        } else {
                            println("This GEMM case is not programmed: transpose_C {}, transpose_A {}, transpose_B {}", transpose_C,
                                    transpose_A, transpose_B);
                            std::abort();
                        }
                    }
                }
            }
            // If we make it here, then none of our algorithms for this last block could be used.
            // Fall through to the generic algorithm below.
        } while (false);
    }

    // If we somehow make it here, then none of our algorithms above could be used. Attempt to use
    // the generic algorithm instead.
    einsum_generic_algorithm<T>(C_indices, A_indices, B_indices, target_dims, link_dims, target_position_in_C, link_position_in_link,
                                C_prefactor, C, AB_prefactor, A, B);
}

} // namespace Detail

template <template <size_t, typename> typename AType, size_t ARank, template <size_t, typename> typename BType, size_t BRank,
          template <size_t, typename> typename CType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices,
          typename U, typename T = double>
auto einsum(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<CRank, T> *C, const U UAB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType<ARank, T> &A, const std::tuple<BIndices...> &B_indices,
            const BType<BRank, T> &B)
    -> std::enable_if_t<std::is_base_of_v<::einsums::Detail::TensorBase<ARank, T>, AType<ARank, T>> &&
                        std::is_base_of_v<::einsums::Detail::TensorBase<BRank, T>, BType<BRank, T>> &&
                        std::is_base_of_v<::einsums::Detail::TensorBase<CRank, T>, CType<CRank, T>> && std::is_arithmetic_v<U>> {
    Timer::push(fmt::format(R"(einsum: "{}"{} = {} "{}"{} * "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UAB_prefactor,
                            A.name(), print_tuple_no_type(A_indices), B.name(), print_tuple_no_type(B_indices), UC_prefactor, C->name(),
                            print_tuple_no_type(C_indices)));

    const T C_prefactor = UC_prefactor;
    const T AB_prefactor = UAB_prefactor;

#if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    // Clone C into a new tensor
    Tensor<CRank, T> testC{C->dims()};
    testC = *C;

    // Perform the einsum using only the generic algorithm
    Timer::push("testing");
    Detail::einsum<true>(C_prefactor, C_indices, &testC, AB_prefactor, A_indices, A, B_indices, B);
    Timer::pop();
#endif

    // Perform the actual einsum
    Detail::einsum<false>(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, B);

#if defined(EINSUMS_CONTINUOUSLY_TEST_EINSUM)
    if constexpr (CRank != 0) {
        // Need to walk through the entire C and testC comparing values and reporting differences.
        auto target_dims = get_dim_ranges<CRank>(*C);

        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            const double Cvalue = std::apply(*C, target_combination);
            const double Ctest = std::apply(testC, target_combination);

            if (std::fabs(Cvalue - Ctest) > 1.0E-6) {
                println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "    !!! EINSUM ERROR !!!");
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
                println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);

                println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ({:})", print_tuple_no_type(target_combination));
                println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} C({:}) += {:f} A({:}) * B({:})", C_prefactor,
                        print_tuple_no_type(C_indices), AB_prefactor, print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

#if defined(EINSUMS_TEST_EINSUM_ABORT)
                std::abort();
#endif
            }
        }
    } else {
        const double Cvalue = *C;
        const double Ctest = testC;

        if (std::fabs(Cvalue - testC) > 1.0E-6) {
            println(emphasis::bold | bg(fmt::color::red) | fg(fmt::color::white), "!!! EINSUM ERROR !!!");
            println(bg(fmt::color::red) | fg(fmt::color::white), "    Expected {:20.14f}", Ctest);
            println(bg(fmt::color::red) | fg(fmt::color::white), "    Obtained {:20.14f}", Cvalue);

            println(bg(fmt::color::red) | fg(fmt::color::white), "    tensor element ()");
            println(bg(fmt::color::red) | fg(fmt::color::white), "    {:f} C() += {:f} A({:}) * B({:})", C_prefactor, AB_prefactor,
                    print_tuple_no_type(A_indices), print_tuple_no_type(B_indices));

#if defined(EINSUMS_TEST_EINSUM_ABORT)
            std::abort();
#endif
        }
    }
#endif
    Timer::pop();
}

// Einsums with provided prefactors.
// 1. C n A n B n is defined above as the base implementation.

// 2. C n A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, A, B_indices, *B);
}

// 3. C n A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && ~is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, B);
}

// 4. C n A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C, AB_prefactor, A_indices, *A, B_indices, *B);
}

// 5. C y A n B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, B);
}

// 6. C y A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, A, B_indices, *B);
}

// 7. C y A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, B);
}

// 8. C y A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices, typename T>
auto einsum(const T C_prefactor, const std::tuple<CIndices...> &C_indices, CType *C, const T AB_prefactor,
            const std::tuple<AIndices...> &A_indices, const AType &A, const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(C_prefactor, C_indices, C->get(), AB_prefactor, A_indices, *A, B_indices, *B);
}

//
// Einsums with default prefactors.
//

// 1. C n A n B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, B);
}

// 2. C n A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, A, B_indices, *B);
}

// 3. C n A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, B);
}

// 4. C n A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<!is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C, 1, A_indices, *A, B_indices, *B);
}

// 5. C y A n B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, B);
}

// 6. C y A n B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && !is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, A, B_indices, *B);
}

// 7. C y A y B n
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && !is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, B);
}

// 8. C y A y B y
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, CType *C, const std::tuple<AIndices...> &A_indices, const AType &A,
            const std::tuple<BIndices...> &B_indices, const BType &B)
    -> std::enable_if_t<is_smart_pointer_v<CType> && is_smart_pointer_v<AType> && is_smart_pointer_v<BType>> {
    einsum(0, C_indices, C->get(), 1, A_indices, *A, B_indices, *B);
}

//
// sort algorithm
//
template <template <size_t, typename> typename AType, size_t ARank, template <size_t, typename> typename CType, size_t CRank,
          typename... CIndices, typename... AIndices, typename U, typename T = double>
auto sort(const U UC_prefactor, const std::tuple<CIndices...> &C_indices, CType<CRank, T> *C, const U UA_prefactor,
          const std::tuple<AIndices...> &A_indices, const AType<ARank, T> &A)
    -> std::enable_if_t<std::is_base_of_v<::einsums::Detail::TensorBase<CRank, T>, CType<CRank, T>> &&
                        std::is_base_of_v<::einsums::Detail::TensorBase<ARank, T>, AType<ARank, T>> &&
                        sizeof...(CIndices) == sizeof...(AIndices) && sizeof...(CIndices) == CRank && sizeof...(AIndices) == ARank &&
                        std::is_arithmetic_v<U>> {

    Timer::push(fmt::format(R"(sort: "{}"{} = {} "{}"{} + {} "{}"{})", C->name(), print_tuple_no_type(C_indices), UA_prefactor, A.name(),
                            print_tuple_no_type(A_indices), UC_prefactor, C->name(), print_tuple_no_type(C_indices)));
    const T C_prefactor = UC_prefactor;
    const T A_prefactor = UA_prefactor;

    // Error check:  If there are any remaining indices then we cannot perform a sort
    constexpr auto check = difference_t<std::tuple<AIndices...>, std::tuple<CIndices...>>();
    static_assert(std::tuple_size_v<decltype(check)> == 0);

    auto target_position_in_A = Detail::find_type_with_position(C_indices, A_indices);

    auto target_dims = get_dim_ranges<CRank>(*C);
    auto a_dims = Detail::get_dim_ranges_for(A, target_position_in_A);

    // HPTT interface currently only works for full Tensors and not TensorViews
#if defined(EINSUMS_USE_HPTT)
    if constexpr (std::is_same_v<CType<CRank, T>, Tensor<CRank, T>> && std::is_same_v<AType<ARank, T>, Tensor<ARank, T>>) {
        std::array<int, ARank> perms{};
        std::array<int, ARank> size{};

        for (int i0 = 0; i0 < ARank; i0++) {
            perms[i0] = get_from_tuple<unsigned long>(target_position_in_A, (2 * i0) + 1);
            size[i0] = A.dim(i0);
        }

        auto plan = hptt::create_plan(perms.data(), ARank, A_prefactor, A.data(), size.data(), nullptr, C_prefactor, C->data(), nullptr,
                                      hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
        plan->execute();
    } else
#endif
        if constexpr (std::is_same_v<decltype(A_indices), decltype(C_indices)>) {
        if (C_prefactor != T{1.0})
            LinearAlgebra::scale(C_prefactor, C);
        LinearAlgebra::axpy(A_prefactor, A, C);
    } else {
        auto view = std::apply(ranges::views::cartesian_product, target_dims);
#pragma omp parallel for simd
        for (auto it = view.begin(); it < view.end(); it++) {
            auto A_order = Detail::construct_indices<AIndices...>(*it, target_position_in_A, *it, target_position_in_A);

            T &target_value = std::apply(*C, *it);
            T A_value = std::apply(A, A_order);

            target_value = C_prefactor * target_value + A_prefactor * A_value;
        }
    }
    Timer::pop();
}

// Sort with default values, no smart pointers
template <typename ObjectA, typename ObjectC, typename... CIndices, typename... AIndices>
auto sort(const std::tuple<CIndices...> &C_indices, ObjectC *C, const std::tuple<AIndices...> &A_indices, const ObjectA &A)
    -> std::enable_if_t<!is_smart_pointer_v<ObjectA> && !is_smart_pointer_v<ObjectC>> {
    sort(0, C_indices, C, 1, A_indices, A);
}

// Sort with default values, two smart pointers
template <typename SmartPointerA, typename SmartPointerC, typename... CIndices, typename... AIndices>
auto sort(const std::tuple<CIndices...> &C_indices, SmartPointerC *C, const std::tuple<AIndices...> &A_indices, const SmartPointerA &A)
    -> std::enable_if_t<is_smart_pointer_v<SmartPointerA> && is_smart_pointer_v<SmartPointerC>> {
    sort(0, C_indices, C->get(), 1, A_indices, *A);
}

// Sort with default values, one smart pointer (A)
template <typename SmartPointerA, typename PointerC, typename... CIndices, typename... AIndices>
auto sort(const std::tuple<CIndices...> &C_indices, PointerC *C, const std::tuple<AIndices...> &A_indices, const SmartPointerA &A)
    -> std::enable_if_t<is_smart_pointer_v<SmartPointerA> && !is_smart_pointer_v<PointerC>> {
    sort(0, C_indices, C, 1, A_indices, *A);
}

// Sort with default values, one smart pointer (C)
template <typename ObjectA, typename SmartPointerC, typename... CIndices, typename... AIndices>
auto sort(const std::tuple<CIndices...> &C_indices, SmartPointerC *C, const std::tuple<AIndices...> &A_indices, const ObjectA &A)
    -> std::enable_if_t<!is_smart_pointer_v<ObjectA> && is_smart_pointer_v<SmartPointerC>> {
    sort(0, C_indices, C->get(), 1, A_indices, A);
}

//
// Element Transform
///

template <template <size_t, typename> typename CType, size_t CRank, typename UnaryOperator, typename T = double>
auto element_transform(CType<CRank, T> *C, UnaryOperator unary_opt)
    -> std::enable_if_t<std::is_base_of_v<::einsums::Detail::TensorBase<CRank, T>, CType<CRank, T>>> {
    Timer::push(fmt::format("element transform: {}", C->name()));
    auto target_dims = get_dim_ranges<CRank>(*C);
    auto view = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for simd
    for (auto it = view.begin(); it != view.end(); it++) {
        T &target_value = std::apply(*C, *it);
        target_value = unary_opt(target_value);
    }
    Timer::pop();
}

template <typename SmartPtr, typename UnaryOperator>
auto element_transform(SmartPtr *C, UnaryOperator unary_opt) -> std::enable_if_t<is_smart_pointer_v<SmartPtr>> {
    element_transform(C->get(), unary_opt);
}

template <template <size_t, typename> typename CType, template <size_t, typename> typename... MultiTensors, size_t Rank,
          typename MultiOperator, typename T = double>
auto element(MultiOperator multi_opt, CType<Rank, T> *C, MultiTensors<Rank, T>... tensors) {
    Timer::push("element");
    auto target_dims = get_dim_ranges<Rank>(*C);
    auto view = std::apply(ranges::views::cartesian_product, target_dims);

    // Ensure the various tensors passed in are the same dimensionality
    if (((C->dims() != tensors.dims()) || ...)) {
        println_abort("element: at least one tensor does not have same dimensionality as destination");
    }

#pragma omp parallel for simd
    for (auto it = view.begin(); it != view.end(); it++) {
        T &target_value = std::apply(*C, *it);
        target_value = multi_opt(target_value, std::apply(tensors, *it)...);
    }
    Timer::pop();
}

} // namespace einsums::TensorAlgebra