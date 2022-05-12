#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

#include <cmath>
#include <functional>

namespace einsums::decomposition {

template <size_t TRank>
auto validate_cp_rank(const Dim<TRank> shape, const std::string &rounding = "round") -> size_t {
    using rounding_func_t = double (*)(double);
    rounding_func_t rounding_func;

    if (rounding == "ceil") {
        rounding_func = ::ceil;
    } else if (rounding == "floor") {
        rounding_func = ::floor;
    } else if (rounding == "round") {
        rounding_func = ::round;
    } else {
        throw std::runtime_error(fmt::format("Rounding should of round, floow, or ceil, but got {}", rounding));
    }

    double prod = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<double>());
    double sum = std::accumulate(shape.begin(), shape.end(), 0, std::plus<double>());

    return static_cast<int>(rounding_func(prod / sum));
}

template <template <size_t, typename> typename TTensor, size_t TRank, typename TType = double>
auto initialize_cp(const TTensor<TRank, TType> &tensor, size_t rank) -> std::vector<Tensor<2, TType>> {

    std::vector<Tensor<2, TType>> factors;

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        auto [U, S, _] = linear_algebra::svd_a(tensor_algebra::unfold<i>(tensor));

        // println(tensor_algebra::unfold<i>(tensor));
        // println(S);

        if (tensor.dim(i) < rank) {
            println_warn("dimension {} size {} is less than the requested decomposition rank {}", i, tensor.dim(i), rank);
            // TODO: Need to padd U up to rank
        }

        // Need to save the factors
        factors.emplace_back(Tensor{U(All{}, Range{0, rank})});

        println(factors[i]);
    });

    return factors;
}

/**
 * CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights; factor[0], ..., factors[-1] |].
 */
template <template <size_t, typename> typename TTensor, size_t TRank, typename TType = double>
auto parafac(const Tensor<TRank, TType> &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8) {
}

} // namespace einsums::decomposition