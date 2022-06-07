#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

#include <cmath>
#include <functional>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::tensor_algebra::index;

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
auto parafac(const TTensor<TRank, TType> &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8) -> std::vector<Tensor<2, TType>> {

    // Perform SVD guess for parafac decomposition procedure
    auto factors = initialize_cp(tensor, rank);

    // Get set of unfolded matrices
    std::vector<Tensor<2, TType>> unfolded_matrices;
    for_sequence<TRank>([&](auto i) {
        unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor));
    });

    int iter = 0;
    while (iter < n_iter_max) {
        int n_ind = 0;
        for_sequence<TRank>([&](auto n) {
            // Form V and Khatri-Rao product intermediates
            Tensor<2, TType> V;
            std::vector<Tensor<2, TType>> KRs;
            bool first = true;

            int m_ind = 0;
            for_sequence<TRank>([&](auto m) {
                if (m_ind != n_ind) {
                    Tensor<2, TType> A_tA{"V", rank, rank};
                    // A_tA = A^T[j] @ A[j]
                    einsum(0.0, Indices{r, s}, &A_tA, 
                           1.0, Indices{I, r}, factors[m_ind], Indices{I, s}, factors[m_ind]);

                    if (first) {
                        V = A_tA;
                        KRs.push_back(factors[m_ind]);
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<2, TType> Vcopy = V;
                        einsum(0.0, Indices{r, s}, &V,
                               1.0, Indices{r, s}, Vcopy, Indices{r, s}, A_tA);
                        
                        // Perform a Khatri-Rao contraction
                        // TODO: Implement an actual Khatri-Rao procedure to replace this "hacky" workaround
                        Tensor<2, TType>& oldKR = KRs.back();

                        size_t running_dim = oldKR.dim(0);
                        size_t appended_dim = tensor.dim(m_ind);
                        Tensor<3, TType> KRtemp{"KRtemp", running_dim, appended_dim, rank};

                        einsum(0.0, Indices{I, M, r}, &KRtemp,
                               1.0, Indices{I, r}, oldKR, Indices{M, r}, factors[m_ind]);

                        // TensorView<2, TType> KRview{KRtemp, Dim<2>{running_dim * appended_dim, rank}};
                        // KR = KRview;

                        Tensor<2, TType> newKR{"New KR", running_dim * appended_dim, rank};

                        for (size_t I = 0; I < running_dim; I++) {
                            for (size_t M = 0; M < appended_dim; M++) {
                                for (size_t r = 0; r < rank; r++) {
                                    newKR(I * appended_dim + M, r) = KRtemp(I, M, r);
                                }
                            }
                        }
                        KRs.push_back(newKR);
                    }
                }
                m_ind += 1;
            });

            /*
            // Update factors[n]
            size_t ndim = tensor.dim(n_ind);

            Tensor<2, TType> fcopy{"fcopy", rank, ndim};

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{r, I}, &fcopy,
                   1.0, Indices{I, K}, unfolded_matrices[n_ind], Indices{K, r}, KRs[KRs.size() - 1]);

            // Step 2: Linear Solving (instead of inversion, for numerical stability)
            linear_algebra::gesv(&V, &fcopy);

            Tensor<2, TType> ones{"ones", rank, ndim};
            ones.set_all(1.0);

            // factors[n] = fcopy^T
            einsum(0.0, Indices{I, r}, &factors[n_ind], 
                   1.0, Indices{r, I}, fcopy, Indices{r, I}, ones);

            println(factors[n_ind]);
            */
            size_t ndim = tensor.dim(n_ind);
            Tensor<2, TType> fcopy{"fcopy", ndim, rank};

            einsum(0.0, Indices{I, r}, &fcopy,
                   1.0, Indices{I, K}, unfolded_matrices[n_ind], Indices{K, r}, KRs.back());
            
            linear_algebra::invert(&V);

            einsum(0.0, Indices{I, s}, &factors[n_ind],
                   1.0, Indices{I, r}, fcopy, Indices{r, s}, V);
            
            n_ind += 1;
        });
        iter += 1;
    }

    // Return **non-normalized** factors
    // TODO: Normalize :)
    return factors;

}

} // namespace einsums::decomposition