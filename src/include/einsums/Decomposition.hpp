#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/Utilities.hpp"

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

/**
 * Computes the 2-norm of a tensor
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto norm(const TTensor<TType, TRank> &tensor) -> TType {
    TType val = 0.0;
    auto target_dims = get_dim_ranges<TRank>(tensor);

    for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
        TType target = std::apply(tensor, target_combination);
        val += target * target;
    }

    return std::sqrt(val);
}

/**
 * Computes the RMSD between two tensors of arbitrary dimension
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto rmsd(const TTensor<TType, TRank> &tensor1, const TTensor<TType, TRank> &tensor2) -> TType {
    TType diff = 0.0;
    auto target_dims = get_dim_ranges<TRank>(tensor1);

    size_t nelem = 0;
    for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
        TType target1 = std::apply(tensor1, target_combination);
        TType target2 = std::apply(tensor2, target_combination);
        diff += (target1 - target2) * (target1 - target2);
        nelem += 1;
    }

    return std::sqrt(diff / nelem);
}

/**
 * Reconstructs a tensor given a CANDECOMP/PARAFAC decomposition
 *
 *   factors = The decomposed CANDECOMP matrices (dimension: [dim[i], rank])
 */
template <size_t TRank, typename TType>
auto parafac_reconstruct(const std::vector<Tensor<TType, 2>> &factors) -> Tensor<TType, TRank> {
    size_t rank = 0;
    Dim<TRank> dims;

    size_t i = 0;
    for (const auto &factor : factors) {
        dims[i] = factor.dim(0);
        if (!rank)
            rank = factor.dim(1);
        i++;
    }

    Tensor<TType, TRank> new_tensor(dims);
    new_tensor.zero();

    auto indices = get_dim_ranges<TRank>(new_tensor);

    for (auto idx_combo : std::apply(ranges::views::cartesian_product, indices)) {
        TType &target = std::apply(new_tensor, idx_combo);
        for (size_t r = 0; r < rank; r++) {
            double temp = 1.0;
            for_sequence<TRank>([&](auto n) { temp *= factors[n](std::get<n>(idx_combo), r); });
            target += temp;
        }
    }

    return new_tensor;
}

template <size_t TRank, typename TType>
auto initialize_cp(std::vector<Tensor<TType, 2>> &folds, size_t rank) -> std::vector<Tensor<TType, 2>> {

    std::vector<Tensor<TType, 2>> factors;

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        auto nthread = omp_get_num_threads();
        omp_set_num_threads(1);
        auto [U, S, _] = linear_algebra::svd_a(folds[i]);
        omp_set_num_threads(nthread);

        // println(tensor_algebra::unfold<i>(tensor));
        // println(S);
        // println(U);
        // println(S);

        // If (i == 0), Scale U by the singular values
        if (i == 0) {
            for (size_t c = 0; c < U.dim(0); c++) {
                double scaling_factor = 0.0;
                if (c < S.dim(0))
                    scaling_factor = S(c);
                linear_algebra::scale_column(c, S(c), &U);
            }
        }

        // println("After scaling");
        // println(U);

        if (folds[i].dim(0) < rank) {
            // println_warn("dimension {} size {} is less than the requested decomposition rank {}", i, folds[i].dim(0), rank);
            // TODO: Need to padd U up to rank
            Tensor<TType, 2> Unew = create_random_tensor("Padded SVD Left Vectors", folds[i].dim(0), rank);
            Unew(All, Range{0, folds[i].dim(0)}) = U(All, Range{0, folds[i].dim(0)});

            // Need to save the factors
            factors.push_back(Unew);
        } else {
            // Need to save the factors
            factors.emplace_back(Tensor{U(All, Range{0, rank})});
        }

        // println("latest factor added");
        // println(factors[factors.size() - 1]);
        // Tensor<TType, 2> Unew = create_random_tensor("Padded SVD Left Vectors", folds[i].dim(0), rank);
        // factors.emplace_back(Unew);
    });

    return factors;
}

/**
 * CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights; factor[0], ..., factors[-1] |].
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto parafac(const TTensor<TType, TRank> &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8)
    -> std::vector<Tensor<TType, 2>> {

    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;
    using vector = std::vector<TType, AlignedAllocator<TType, 64>>;

    // Compute set of unfolded matrices
    std::vector<Tensor<TType, 2>> unfolded_matrices;
    for_sequence<TRank>([&](auto i) { unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor)); });

    // Perform SVD guess for parafac decomposition procedure
    std::vector<Tensor<TType, 2>> factors = initialize_cp<TRank>(unfolded_matrices, rank);

    double tensor_norm = norm(tensor);
    size_t nelem = 1;
    for_sequence<TRank>([&](auto i) { nelem *= tensor.dim(i); });
    tensor_norm /= std::sqrt((double)nelem);

    int iter = 0;
    bool converged = false;
    double prev_error = 0.0;
    while (iter < n_iter_max) {
        for_sequence<TRank>([&](auto n_ind) {
            // Form V and Khatri-Rao product intermediates
            Tensor<TType, 2> V;
            Tensor<TType, 2> *KR;
            bool first = true;

            for_sequence<TRank>([&](auto m_ind) {
                if (m_ind != n_ind) {
                    Tensor<TType, 2> A_tA{"V", rank, rank};
                    // A_tA = A^T[j] @ A[j]
                    // println("iter {}, mind {}", iter, m_ind);
                    // println(factors[m_ind]);
                    einsum(0.0, Indices{r, s}, &A_tA, 1.0, Indices{I, r}, factors[m_ind], Indices{I, s}, factors[m_ind]);

                    if (first) {
                        V = A_tA;
                        auto *KRcopy = new Tensor<TType, 2>("KR product", tensor.dim(m_ind), rank);
                        *KRcopy = factors[m_ind];
                        KR = KRcopy;
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<TType, 2> Vcopy = V;
                        einsum(0.0, Indices{r, s}, &V, 1.0, Indices{r, s}, Vcopy, Indices{r, s}, A_tA);

                        // Perform a Khatri-Rao contraction
                        // TODO: Implement an actual Khatri-Rao procedure to replace this "hacky" workaround

                        size_t running_dim = KR->dim(0);
                        size_t appended_dim = tensor.dim(m_ind);

                        Tensor<TType, 3> KRbuff{"KR temp", running_dim, appended_dim, rank};

                        einsum(0.0, Indices{I, M, r}, &KRbuff, 1.0, Indices{I, r}, *KR, Indices{M, r}, factors[m_ind]);

                        auto *newKR = new Tensor<TType, 2>("KR product", running_dim * appended_dim, rank);

                        const vector &KRbuffd = KRbuff.vector_data();
                        vector &newKRd = newKR->vector_data();

                        std::copy(KRbuffd.begin(), KRbuffd.end(), newKRd.begin());

                        /*
                        for (size_t I = 0; I < running_dim; I++) {
                            for (size_t M = 0; M < appended_dim; M++) {
                                for (size_t R = 0; R < rank; R++) {
                                    (*newKR)(I * appended_dim + M, R) += (*KR)(I, R) * factors[m_ind](M, R);
                                }
                            }
                        }
                        */

                        delete KR;
                        KR = newKR;
                    }
                }
            });

            // Update factors[n_ind]
            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{I, r}, &factors[n_ind], 1.0, Indices{I, K}, unfolded_matrices[n_ind], Indices{K, r}, *KR);
            delete KR;

            // Step 2: Linear Solve (instead of inversion, for numerical stability, column-major ordering)
            linear_algebra::gesv(&V, &factors[n_ind]);
        });

        // Check for convergence
        // Reconstruct Tensor based on the factors
        Tensor<TType, TRank> rec_tensor = parafac_reconstruct<TRank>(factors);

        double unnormalized_error = rmsd(rec_tensor, tensor);
        double curr_error = unnormalized_error / tensor_norm;
        double delta = std::abs(curr_error - prev_error);

        // printf("    @CP Iteration %d, ERROR: %8.8f, DELTA: %8.8f\n", iter, curr_error, delta);

        if (iter >= 2 && delta < tolerance) {
            converged = true;
            break;
        }

        prev_error = curr_error;
        iter += 1;
    }
    if (!converged) {
        println_warn("CP decomposition failed to converge in {} iterations", n_iter_max);
    }

    // Return **non-normalized** factors
    return factors;
}

template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto tucker_reconstruct(const TTensor<TType, TRank> &g_tensor, const std::vector<TTensor<TType, 2>> &factors) {

    // Dimension workspace for temps
    Dim<TRank> dims_buffer = g_tensor.dims();
    // Buffers to hold intermediates while rebuilding the tensor
    Tensor<TType, TRank> *old_tensor_buffer;
    Tensor<TType, TRank> *new_tensor_buffer;

    old_tensor_buffer = new Tensor(dims_buffer);
    *old_tensor_buffer = g_tensor;

    // Reform the tensor (with all its intermediates)
    for_sequence<TRank>([&](auto i) {
        size_t full_idx = factors[i].dim(0);
        dims_buffer[i] = full_idx;
        new_tensor_buffer = new Tensor(dims_buffer);
        new_tensor_buffer->zero();

        auto source_dims = get_dim_ranges<TRank>(*old_tensor_buffer);

        for (auto source_combination : std::apply(ranges::views::cartesian_product, source_dims)) {
            for (size_t n = 0; n < full_idx; n++) {
                auto target_combination = source_combination;
                std::get<i>(target_combination) = n;

                TType &source = std::apply(*old_tensor_buffer, source_combination);
                TType &target = std::apply(*new_tensor_buffer, target_combination);

                target += source * factors[i](n, std::get<i>(source_combination));
            }
        }

        delete old_tensor_buffer;
        old_tensor_buffer = new_tensor_buffer;
    });

    Tensor<TType, TRank> new_tensor = *(new_tensor_buffer);
    // Only delete one of the buffers, to avoid a double free
    delete new_tensor_buffer;

    return new_tensor;
}

template <size_t TRank, typename TType = double>
auto initialize_tucker(std::vector<Tensor<TType, 2>> &folds, std::vector<size_t> &ranks) -> std::vector<Tensor<TType, 2>> {

    std::vector<Tensor<TType, 2>> factors;

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        size_t rank = ranks[i];
        auto [U, S, _] = linear_algebra::svd_a(folds[i]);

        // println(tensor_algebra::unfold<i>(tensor));
        // println(S);

        if (folds[i].dim(0) < rank) {
            println_warn("dimension {} size {} is less than the requested decomposition rank {}", i, folds[i].dim(0), rank);
            // TODO: Need to padd U up to rank
        }

        // Need to save the factors
        factors.emplace_back(Tensor{U(All, Range{0, rank})});
    });

    return factors;
}

/**
 * Tucker decomposition of a tensor via Higher-Order SVD (HO-SVD).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights[r0][r1]...; factor[0][r0], ..., factors[-1][rn] |]
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto tucker_ho_svd(const TTensor<TType, TRank> &tensor, std::vector<size_t> &ranks,
                   const std::vector<Tensor<TType, 2>> &folds = std::vector<Tensor<TType, 2>>())
    -> std::tuple<Tensor<TType, TRank>, std::vector<Tensor<TType, 2>>> {

    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // Compute set of unfolded matrices
    std::vector<Tensor<TType, 2>> unfolded_matrices;
    if (!folds.size()) {
        for_sequence<TRank>([&](auto i) { unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor)); });
    } else {
        unfolded_matrices = folds;
    }

    // Perform SVD guess for tucker decomposition procedure
    std::vector<Tensor<TType, 2>> factors = initialize_tucker<TRank>(unfolded_matrices, ranks);

    // Get the dimension workspace for temps
    Dim<TRank> dims_buffer = tensor.dims();
    // Make buffers to hold intermediates while forming G
    Tensor<TType, TRank> *old_g_buffer;
    Tensor<TType, TRank> *new_g_buffer;

    old_g_buffer = new Tensor(dims_buffer);
    *old_g_buffer = tensor;

    // Form G (with all of its intermediates)
    for_sequence<TRank>([&](auto i) {
        size_t rank = ranks[i];
        dims_buffer[i] = rank;
        new_g_buffer = new Tensor(dims_buffer);
        new_g_buffer->zero();

        auto source_dims = get_dim_ranges<TRank>(*old_g_buffer);

        for (auto source_combination : std::apply(ranges::views::cartesian_product, source_dims)) {
            for (size_t r = 0; r < rank; r++) {
                auto target_combination = source_combination;
                std::get<i>(target_combination) = r;

                TType &source = std::apply(*old_g_buffer, source_combination);
                TType &target = std::apply(*new_g_buffer, target_combination);

                target += source * factors[i](std::get<i>(source_combination), r);
            }
        }

        delete old_g_buffer;
        old_g_buffer = new_g_buffer;
    });

    Tensor<TType, TRank> g_tensor = *(new_g_buffer);
    // ONLY delete one of the buffers, to avoid a double free
    delete new_g_buffer;

    return std::make_tuple(g_tensor, factors);
}

/**
 * Tucker decomposition via Higher-Order Orthogonal Inversion (HO-OI).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights[r0][r1]...; factor[0][r1], ..., factors[-1][rn] |].
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto tucker_ho_oi(const TTensor<TType, TRank> &tensor, std::vector<size_t> &ranks, int n_iter_max = 100, double tolerance = 1.e-8)
    -> std::tuple<TTensor<TType, TRank>, std::vector<Tensor<TType, 2>>> {

    // Use HO SVD as a starting guess
    auto ho_svd_guess = tucker_ho_svd(tensor, ranks);
    auto g_tensor = std::get<0>(ho_svd_guess);
    auto factors = std::get<1>(ho_svd_guess);

    int iter = 0;
    bool converged = false;
    while (iter < n_iter_max) {
        std::vector<Tensor<TType, 2>> new_folds;

        for_sequence<TRank>([&](auto i) {
            // Make the workspace for the contraction
            Dim<TRank> dims_buffer = tensor.dims();
            // Make buffers to form intermediates while forming new folds
            Tensor<TType, TRank> *old_fold_buffer;
            Tensor<TType, TRank> *new_fold_buffer;

            // Initialize old fold buffer to the tensor
            old_fold_buffer = new Tensor(dims_buffer);
            *old_fold_buffer = tensor;

            for_sequence<TRank>([&](auto j) {
                if (j != i) {
                    size_t rank = ranks[j];
                    dims_buffer[j] = rank;
                    new_fold_buffer = new Tensor(dims_buffer);
                    new_fold_buffer->zero();

                    auto source_dims = get_dim_ranges<TRank>(*old_fold_buffer);

                    for (auto source_combination : std::apply(ranges::views::cartesian_product, source_dims)) {
                        for (size_t r = 0; r < rank; r++) {
                            auto target_combination = source_combination;
                            std::get<j>(target_combination) = r;

                            TType &source = std::apply(*old_fold_buffer, source_combination);
                            TType &target = std::apply(*new_fold_buffer, target_combination);

                            target += source * factors[j](std::get<j>(source_combination), r);
                        }
                    }

                    delete old_fold_buffer;
                    old_fold_buffer = new_fold_buffer;
                }
            });

            Tensor<TType, 2> new_fold = tensor_algebra::unfold<i>(*new_fold_buffer);
            new_folds.push_back(new_fold);

            // Only delete once to avoid a double free
            delete new_fold_buffer;
        });

        // Reformulate guess based on HO SVD of new_folds
        auto new_ho_svd = tucker_ho_svd(tensor, ranks, new_folds);
        auto new_g_tensor = std::get<0>(new_ho_svd);
        auto new_factors = std::get<1>(new_ho_svd);

        // Check for convergence
        double rmsd_max = rmsd(new_g_tensor, g_tensor);
        for_sequence<TRank>([&](auto n) { rmsd_max = std::max(rmsd_max, rmsd(new_factors[n], factors[n])); });

        // Update G and factors
        g_tensor = new_g_tensor;
        factors = new_factors;

        if (rmsd_max < tolerance) {
            converged = true;
            break;
        }

        iter += 1;
    }
    if (!converged) {
        println_warn("Tucker HO-OI decomposition failed to converge in {} iterations", n_iter_max);
    }

    return std::make_tuple(g_tensor, factors);
}

} // namespace einsums::decomposition