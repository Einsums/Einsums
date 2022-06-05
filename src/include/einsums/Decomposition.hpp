#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

#include <cmath>
#include <functional>

using einsums;
using einsums::TensorAlgebra;
using einsums::TensorAlgebra::Index;

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
auto parafac(const Tensor<TRank, TType> &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8) -> std::vector<Tensor<2, TType>> {

    // Get the sizes of each dimension of the tensor
    auto tensor_dim_ranges = get_dim_ranges(tensor);
    size_t t_rank = tensor_dim_ranges.size();

    // Perform SVD guess for parafac decomposition procedure
    auto factors = initialize_cp(tensor, rank);

    // Get set of unfolded matrices
    std::vector<Tensor<2, TType>> unfolded_matrices;
    for_sequence<TRank>([&](auto i) {
        unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor));
    });

    int iter = 0;
    while (iter < n_iter_max) {
        for_sequence<TRank>([&](auto n) {
            // Form V intermediate
            Tensor<2, TType> V;
            bool first = true;

            for_sequence<TRank>([&](auto m) {
                if (m == n) continue;

                Tensor<2, Type> A_tA{"V", rank, rank};
                // A_tA = A^T[j] @ A[j]
                einsum(0.0, Indices{r, s}, &A_tA, 
                       1.0, Indices{I, r}, factors[j], Indices{I, s}, factors[m]);

                if (first) {
                    V = A_tA;
                    first = false;
                } else {
                    Tensor<2, Ttype> Vcopy = V;
                    einsum(0.0, Indices{r, s}, &V,
                           1.0, Indices{r, s}, Vcopy, Indices{r, s}, A_tA);
                }
            });

            // Form Khatri-Rao Intermediate
            Tensor<2, TType> KR;
            first = true;

            for_sequence<TRank>([&](auto j) {
                size_t m = t_rank - j - 1;
                if (m == n) continue;

                if (first) {
                    KR = factors[m];
                    first = false;
                } else {
                    // Perform a Khatri-Rao contraction
                    // TODO: Implement an actual Khatri-Rao procedure to replace this "hacky" workaround
                    size_t running_dim = get_dim_ranges(KR)[0];
                    size_t appended_dim = tensor_dim_ranges[m];
                    Tensor<3, TType> KRtemp{"KRtemp", running_dim, appended_dim, rank};

                    einsum(0.0, Indices{I, J, r}, &KRtemp,
                           1.0, Indices{I, r}, KR, Indices{J, r}, factors[m]);

                    TensorView<2, TType> KRview{KRtemp, {running_dim * appended_dim, rank}};
                    KR = KRview;
                }
            });

            // Update factors[i]
            size_t idim = tensor_dim_ranges[i];

            Tensor<2, Ttype> fcopy{"fcopy", rank, idim};

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{r, I}, &fcopy,
                   1.0, Indices{I, K}, unfolded_matrices[i], Indices{K, r}, KR);

            // Step 2: Linear Solving (instead of inversion, for numerical stability)
            linear_algebra::gesv(&V, &fcopy);

            // Transpose the factors to get the right form
            TensorView<2, TType> fview{fcopy, Dim<Rank>{idim, rank}};
            factors[i] = fview;
        });
    }

    // Return **non-normalized** factors
    // TODO: Normalize :)
    return factors;

}

} // namespace einsums::decomposition