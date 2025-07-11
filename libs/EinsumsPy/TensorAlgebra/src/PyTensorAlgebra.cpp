//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include <EinsumsPy/TensorAlgebra/PyTensorAlgebra.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <EinsumsPy/GPU/PyGPUView.hpp>
using namespace einsums::python;
#endif

#include <deque>

using namespace std;
using namespace einsums;
using namespace einsums::tensor_algebra;

namespace py = pybind11;

string einsums::tensor_algebra::detail::intersect(string const &st1, string const &st2) {
    string out = "";

    for (int i = 0; i < st1.length(); i++) {
        for (int j = 0; j < st2.length(); j++) {
            if (st1[i] == st2[j]) {
                out += st1[i];
            }
        }
    }
    return out;
}

PyEinsumGenericPlan::PyEinsumGenericPlan(int num_inds, std::vector<int> C_permute, std::vector<int> A_permute, std::vector<int> B_permute)
    : _num_inds{num_inds}, _C_permute{C_permute}, _A_permute{A_permute}, _B_permute{B_permute},
      _direct_product_swap{(_A_permute.size() == _B_permute.size()) && (_A_permute.size() == _C_permute.size()) &&
                           (detail::intersect(_A_permute, _B_permute).size() == _A_permute.size()) &&
                           (detail::intersect(_A_permute, _C_permute).size() == _A_permute.size()) &&
                           (detail::intersect(_B_permute, _C_permute).size() == _A_permute.size())} {
}

PyEinsumDotPlan::PyEinsumDotPlan(PyEinsumGenericPlan const &plan_base) : PyEinsumGenericPlan(plan_base) {
}

PyEinsumDirectProductPlan::PyEinsumDirectProductPlan(PyEinsumGenericPlan const &plan_base) : PyEinsumGenericPlan(plan_base) {
}

PyEinsumGerPlan::PyEinsumGerPlan(std::vector<int> const &CA_target_pos, std::vector<int> const &CB_target_pos, bool swap_AB,
                                 PyEinsumGenericPlan const &plan_base)
    : PyEinsumGenericPlan(plan_base), _CA_target_pos{CA_target_pos}, _CB_target_pos{CB_target_pos}, _swap_AB{swap_AB} {
}

PyEinsumGemvPlan::PyEinsumGemvPlan(std::vector<int> const &A_link_inds, std::vector<int> const &B_link_inds,
                                   std::vector<int> const &AC_inds, int A_target_last_ind, int A_link_last_ind, int B_link_last_ind,
                                   int C_target_last_ind, bool trans_A, bool swap_AB, PyEinsumGenericPlan const &plan_base)
    : PyEinsumGenericPlan(plan_base), _A_link_pos{A_link_inds}, _B_link_pos{B_link_inds}, _AC_pos{AC_inds}, _trans_A{trans_A},
      _A_link_last_ind{A_link_last_ind}, _A_target_last_ind{A_target_last_ind}, _B_link_last_ind{B_link_last_ind},
      _C_target_last_ind{C_target_last_ind}, _swap_AB{swap_AB} {
}

PyEinsumGemmPlan::PyEinsumGemmPlan(std::vector<int> const &A_link_inds, std::vector<int> const &B_link_inds,
                                   std::vector<int> const &AC_inds, std::vector<int> const &BC_inds, int A_target_last_ind,
                                   int A_link_last_ind, int B_target_last_ind, int B_link_last_ind, int CA_target_last_ind,
                                   int CB_target_last_ind, bool trans_A, bool trans_B, bool trans_C, PyEinsumGenericPlan const &plan_base)
    : PyEinsumGenericPlan(plan_base), _A_link_inds{A_link_inds}, _B_link_inds{B_link_inds}, _AC_inds{AC_inds}, _BC_inds{BC_inds},
      _A_target_last_ind{A_target_last_ind}, _A_link_last_ind{A_link_last_ind}, _B_target_last_ind{B_target_last_ind},
      _B_link_last_ind{B_link_last_ind}, _CA_target_last_ind{CA_target_last_ind}, _CB_target_last_ind{CB_target_last_ind},
      _trans_A{trans_A}, _trans_B{trans_B}, _trans_C{trans_C} {
}

static string difference(string const &st1, string const &st2) {
    string out = "";

    for (int i = 0; i < st1.length(); i++) {
        bool add = true;
        for (int j = 0; j < st2.length(); j++) {
            if (st1[i] == st2[j]) {
                add = false;
                break;
            }
        }
        if (add) {
            out += st1[i];
        }
    }
    return out;
}

static string unique(string const &str) {
    string out = "";

    for (int i = 0; i < str.length(); i++) {
        bool add = true;
        for (int j = 0; j < out.length(); j++) {
            if (out[j] == str[i]) {
                add = false;
            }
        }
        if (add) {
            out += str[i];
        }
    }
    return out;
}

static deque<pair<char, size_t>> find_ind_with_pos(string const &st1, string const &st2) {
    deque<pair<char, size_t>> out;

    for (int i = 0; i < st1.length(); i++) {
        for (int j = 0; j < st2.length(); j++) {
            if (st1[i] == st2[j]) {
                out.push_back(pair<char, size_t>(st1[i], j));
            }
        }
    }

    return out;
}

static bool contiguous_indices(deque<pair<char, size_t>> const &pairs) {
    if (pairs.size() == 0) {
        return true;
    }
    for (int i = 0; i < pairs.size() - 1; i++) {
        if (pairs[i].second != pairs[i + 1].second - 1) {
            return false;
        }
    }
    return true;
}

static bool same_ordering(deque<pair<char, size_t>> const &x, deque<pair<char, size_t>> const &y) {
    if (x.size() != y.size()) {
        return false;
    } else if (x.size() == 0 && y.size() == 0) {
        return false;
    }
    for (int i = 0; i < x.size(); i++) {
        if (x[i].first != y[i].first) {
            return false;
        }
    }
    return true;
}

static vector<int> create_perm_table(string const &indices, string const &unique_indices) {
    vector<int> out(indices.size());

    for (int i = 0; i < indices.length(); i++) {
        for (int j = 0; j < unique_indices.length(); j++) {
            if (indices[i] == unique_indices[j]) {
                out[i] = j;
                break;
            }
        }
    }
    return out;
}

std::shared_ptr<einsums::tensor_algebra::PyEinsumGenericPlan>
einsums::tensor_algebra::compile_plan(std::string C_indices, std::string A_indices, std::string B_indices) {
    using namespace einsums::tensor_algebra::detail;
    size_t ARank = A_indices.size();
    size_t BRank = B_indices.size();
    size_t CRank = C_indices.size();

    std::string linksAB = intersect(A_indices, B_indices);

    std::string links = difference(linksAB, C_indices);

    std::string CAlinks = intersect(C_indices, A_indices);

    std::string CBlinks = intersect(C_indices, B_indices);

    std::string CminusA = difference(C_indices, A_indices);

    std::string CminusB = difference(C_indices, B_indices);

    bool have_remaining_indices_in_CminusA = CminusA.size() > 0;
    bool have_remaining_indices_in_CminusB = CminusB.size() > 0;

    std::string A_only = difference(A_indices, links);

    std::string B_only = difference(B_indices, links);

    std::string A_unique = unique(A_indices);

    std::string B_unique = unique(B_indices);

    std::string C_unique = unique(C_indices);

    std::string link_unique = unique(links);

    bool A_hadamard_found = ARank != A_unique.size();
    bool B_hadamard_found = BRank != B_unique.size();
    bool C_hadamard_found = CRank != C_unique.size();

    auto link_position_in_A    = find_ind_with_pos(link_unique, A_indices);
    auto link_position_in_B    = find_ind_with_pos(link_unique, B_indices);
    auto link_position_in_link = find_ind_with_pos(link_unique, links);

    auto target_position_in_A = find_ind_with_pos(C_unique, A_indices);
    auto target_position_in_B = find_ind_with_pos(C_unique, B_indices);
    auto target_position_in_C = find_ind_with_pos(C_unique, C_indices);

    auto A_target_position_in_C = find_ind_with_pos(A_indices, C_indices);
    auto B_target_position_in_C = find_ind_with_pos(B_indices, C_indices);

    bool contiguous_link_position_in_A = contiguous_indices(link_position_in_A);
    bool contiguous_link_position_in_B = contiguous_indices(link_position_in_B);

    bool contiguous_target_position_in_A = contiguous_indices(target_position_in_A);
    bool contiguous_target_position_in_B = contiguous_indices(target_position_in_B);

    bool contiguous_A_targets_in_C = contiguous_indices(A_target_position_in_C);
    bool contiguous_B_targets_in_C = contiguous_indices(B_target_position_in_C);

    bool same_ordering_link_position_in_AB   = same_ordering(link_position_in_A, link_position_in_B);
    bool same_ordering_target_position_in_CA = same_ordering(target_position_in_A, A_target_position_in_C);
    bool same_ordering_target_position_in_CB = same_ordering(target_position_in_B, B_target_position_in_C);

    bool C_exactly_matches_A = (A_indices == C_indices);
    bool C_exactly_matches_B = (B_indices == C_indices);
    bool A_exactly_matches_B = (A_indices == B_indices);

    bool is_gemm_possible = have_remaining_indices_in_CminusA && have_remaining_indices_in_CminusB && contiguous_link_position_in_A &&
                            contiguous_link_position_in_B && contiguous_target_position_in_A && contiguous_target_position_in_B &&
                            contiguous_A_targets_in_C && contiguous_B_targets_in_C && same_ordering_link_position_in_AB &&
                            same_ordering_target_position_in_CA && same_ordering_target_position_in_CB && !A_hadamard_found &&
                            !B_hadamard_found && !C_hadamard_found;
    bool is_gemv_AB_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                               same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                               !same_ordering_target_position_in_CB && B_target_position_in_C.size() == 0 && !A_hadamard_found &&
                               !B_hadamard_found && !C_hadamard_found;
    bool is_gemv_BA_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_B &&
                               same_ordering_link_position_in_AB && same_ordering_target_position_in_CB &&
                               !same_ordering_target_position_in_CA && A_target_position_in_C.size() == 0 && !A_hadamard_found &&
                               !B_hadamard_found && !C_hadamard_found;
    bool is_gemv_possible = is_gemv_AB_possible || is_gemv_BA_possible;
    bool element_wise_multiplication =
        C_exactly_matches_A && C_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    bool dot_product   = C_indices.length() == 0 && A_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    bool outer_product = linksAB.size() == 0 && contiguous_target_position_in_A && contiguous_target_position_in_B && !A_hadamard_found &&
                         !B_hadamard_found && !C_hadamard_found;

    string unique_indices = unique(A_indices + B_indices + C_indices);

    vector<int> A_perm = create_perm_table(A_indices, unique_indices), B_perm = create_perm_table(B_indices, unique_indices),
                C_perm = create_perm_table(C_indices, unique_indices);

    PyEinsumGenericPlan base(unique_indices.length(), C_perm, A_perm, B_perm);

    if (dot_product) {
        return make_shared<PyEinsumDotPlan>(base);
    } else if (element_wise_multiplication) {
        return make_shared<PyEinsumDirectProductPlan>(base);
    } else if (outer_product) {
        std::vector<int> CA_target_pos, CB_target_pos;
        CA_target_pos.reserve(A_target_position_in_C.size());
        CB_target_pos.reserve(B_target_position_in_C.size());
        bool swap_AB = A_target_position_in_C[0].second != 0;

        for (auto const &pair : A_target_position_in_C) {
            bool has_index = false;
            for (int i = 0; i < CA_target_pos.size(); i++) {
                if (CA_target_pos[i] == pair.second) {
                    has_index = true;
                    break;
                }
            }
            if (!has_index) {
                CA_target_pos.push_back((int)pair.second);
            }
        }

        for (auto const &pair : B_target_position_in_C) {
            bool has_index = false;
            for (int i = 0; i < CB_target_pos.size(); i++) {
                if (CB_target_pos[i] == pair.second) {
                    has_index = true;
                    break;
                }
            }
            if (!has_index) {
                CB_target_pos.push_back((int)pair.second);
            }
        }

        return make_shared<PyEinsumGerPlan>(CA_target_pos, CB_target_pos, swap_AB, base);
    } else if (is_gemv_possible) {
        bool swap_AB = is_gemv_BA_possible && !is_gemv_AB_possible;
        if (!swap_AB) {
            std::vector<int> A_link_inds, B_link_inds, AC_inds;
            A_link_inds.reserve(link_position_in_A.size());
            B_link_inds.reserve(link_position_in_B.size());
            AC_inds.reserve(A_target_position_in_C.size());
            int A_target_last_ind = std::max_element(target_position_in_A.cbegin(), target_position_in_A.cend(),
                                                     [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                         return a.second < b.second;
                                                     })
                                        ->second,
                A_link_last_ind = std::max_element(link_position_in_A.cbegin(), link_position_in_A.cend(),
                                                   [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                       return a.second < b.second;
                                                   })
                                      ->second,
                B_link_last_ind = std::max_element(link_position_in_B.cbegin(), link_position_in_B.cend(),
                                                   [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                       return a.second < b.second;
                                                   })
                                      ->second,
                C_target_last_ind = std::max_element(target_position_in_C.cbegin(), target_position_in_C.cend(),
                                                     [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                         return a.second < b.second;
                                                     })
                                        ->second;
            bool trans_A = (link_position_in_A[0].second == 0);

            for (auto const &pair : A_target_position_in_C) {
                bool has_index = false;
                for (int i = 0; i < AC_inds.size(); i++) {
                    if (AC_inds[i] == pair.second) {
                        has_index = true;
                        break;
                    }
                }
                if (!has_index) {
                    AC_inds.push_back((int)pair.second);
                }
            }

            for (auto const &pair : link_position_in_A) {
                bool has_index = false;
                for (int i = 0; i < A_link_inds.size(); i++) {
                    if (A_link_inds[i] == pair.second) {
                        has_index = true;
                        break;
                    }
                }
                if (!has_index) {
                    A_link_inds.push_back((int)pair.second);
                }
            }

            for (auto const &pair : link_position_in_B) {
                bool has_index = false;
                for (int i = 0; i < B_link_inds.size(); i++) {
                    if (B_link_inds[i] == pair.second) {
                        has_index = true;
                        break;
                    }
                }
                if (!has_index) {
                    B_link_inds.push_back((int)pair.second);
                }
            }

            return make_shared<PyEinsumGemvPlan>(A_link_inds, B_link_inds, AC_inds, A_target_last_ind, A_link_last_ind, B_link_last_ind,
                                                 C_target_last_ind, trans_A, swap_AB, base);
        } else {
            std::vector<int> A_link_inds, B_link_inds, AC_inds;
            A_link_inds.reserve(link_position_in_B.size());
            B_link_inds.reserve(link_position_in_A.size());
            AC_inds.reserve(B_target_position_in_C.size());
            int A_target_last_ind = std::max_element(target_position_in_B.cbegin(), target_position_in_B.cend(),
                                                     [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                         return a.second < b.second;
                                                     })
                                        ->second,
                A_link_last_ind = std::max_element(link_position_in_B.cbegin(), link_position_in_B.cend(),
                                                   [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                       return a.second < b.second;
                                                   })
                                      ->second,
                B_link_last_ind = std::max_element(link_position_in_A.cbegin(), link_position_in_A.cend(),
                                                   [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                       return a.second < b.second;
                                                   })
                                      ->second,
                C_target_last_ind = std::max_element(target_position_in_C.cbegin(), target_position_in_C.cend(),
                                                     [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                         return a.second < b.second;
                                                     })
                                        ->second;
            bool trans_A = (link_position_in_B[0].second == 0);

            for (auto const &pair : B_target_position_in_C) {
                bool has_index = false;
                for (int i = 0; i < AC_inds.size(); i++) {
                    if (AC_inds[i] == pair.second) {
                        has_index = true;
                        break;
                    }
                }
                if (!has_index) {
                    AC_inds.push_back((int)pair.second);
                }
            }

            for (auto const &pair : link_position_in_B) {
                bool has_index = false;
                for (int i = 0; i < A_link_inds.size(); i++) {
                    if (A_link_inds[i] == pair.second) {
                        has_index = true;
                        break;
                    }
                }
                if (!has_index) {
                    A_link_inds.push_back((int)pair.second);
                }
            }

            for (auto const &pair : link_position_in_A) {
                bool has_index = false;
                for (int i = 0; i < B_link_inds.size(); i++) {
                    if (B_link_inds[i] == pair.second) {
                        has_index = true;
                        break;
                    }
                }
                if (!has_index) {
                    B_link_inds.push_back((int)pair.second);
                }
            }

            return make_shared<PyEinsumGemvPlan>(A_link_inds, B_link_inds, AC_inds, A_target_last_ind, A_link_last_ind, B_link_last_ind,
                                                 C_target_last_ind, trans_A, swap_AB, base);
        }
    } else if (is_gemm_possible) {
        std::vector<int> A_link_inds, B_link_inds, AC_inds, BC_inds;
        AC_inds.reserve(A_target_position_in_C.size());
        BC_inds.reserve(B_target_position_in_C.size());
        A_link_inds.reserve(link_position_in_A.size());
        B_link_inds.reserve(link_position_in_B.size());
        int A_target_last_ind = std::max_element(target_position_in_A.cbegin(), target_position_in_A.cend(),
                                                 [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                     return a.second < b.second;
                                                 })
                                    ->second,
            A_link_last_ind = std::max_element(link_position_in_A.cbegin(), link_position_in_A.cend(),
                                               [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                   return a.second < b.second;
                                               })
                                  ->second,
            B_target_last_ind = std::max_element(target_position_in_B.cbegin(), target_position_in_B.cend(),
                                                 [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                     return a.second < b.second;
                                                 })
                                    ->second,
            B_link_last_ind = std::max_element(link_position_in_B.cbegin(), link_position_in_B.cend(),
                                               [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                   return a.second < b.second;
                                               })
                                  ->second,
            CA_target_last_ind = std::max_element(A_target_position_in_C.cbegin(), A_target_position_in_C.cend(),
                                                  [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                      return a.second < b.second;
                                                  })
                                     ->second,
            CB_target_last_ind = std::max_element(B_target_position_in_C.cbegin(), B_target_position_in_C.cend(),
                                                  [](std::pair<char, size_t> const &a, std::pair<char, size_t> const &b) -> bool {
                                                      return a.second < b.second;
                                                  })
                                     ->second;
        bool trans_A = (link_position_in_A[0].second == 0), trans_B = (link_position_in_B[0].second != 0),
             trans_C = (A_target_position_in_C[0].second != 0);

        for (auto const &pair : A_target_position_in_C) {
            bool has_index = false;
            for (int i = 0; i < AC_inds.size(); i++) {
                if (AC_inds[i] == pair.second) {
                    has_index = true;
                    break;
                }
            }
            if (!has_index) {
                AC_inds.push_back((int)pair.second);
            }
        }

        for (auto const &pair : B_target_position_in_C) {
            bool has_index = false;
            for (int i = 0; i < BC_inds.size(); i++) {
                if (BC_inds[i] == pair.second) {
                    has_index = true;
                    break;
                }
            }
            if (!has_index) {
                BC_inds.push_back((int)pair.second);
            }
        }

        for (auto const &pair : link_position_in_A) {
            bool has_index = false;
            for (int i = 0; i < A_link_inds.size(); i++) {
                if (A_link_inds[i] == pair.second) {
                    has_index = true;
                    break;
                }
            }
            if (!has_index) {
                A_link_inds.push_back((int)pair.second);
            }
        }

        for (auto const &pair : link_position_in_B) {
            bool has_index = false;
            for (int i = 0; i < B_link_inds.size(); i++) {
                if (B_link_inds[i] == pair.second) {
                    has_index = true;
                    break;
                }
            }
            if (!has_index) {
                B_link_inds.push_back((int)pair.second);
            }
        }

        return make_shared<PyEinsumGemmPlan>(A_link_inds, B_link_inds, AC_inds, BC_inds, A_target_last_ind, A_link_last_ind,
                                             B_target_last_ind, B_link_last_ind, CA_target_last_ind, CB_target_last_ind, trans_A, trans_B,
                                             trans_C, base);
    } else {
        return make_shared<PyEinsumGenericPlan>(base);
    }
}

std::vector<size_t> einsums::tensor_algebra::detail::get_dim_ranges_for_many(pybind11::buffer_info const &C, std::vector<int> const &C_perm,
                                                                             pybind11::buffer_info const &A, std::vector<int> const &A_perm,
                                                                             pybind11::buffer_info const &B, std::vector<int> const &B_perm,
                                                                             int unique_indices) {
    std::vector<size_t> out(unique_indices);
    for (int i = 0; i < unique_indices; i++) {
        out[i] = 0;
    }

    for (int i = 0; i < C_perm.size(); i++) {
        if (out[C_perm[i]] == 0) {
            out[C_perm[i]] = C.shape[i];
        }
    }

    for (int i = 0; i < A_perm.size(); i++) {
        if (out[A_perm[i]] == 0) {
            out[A_perm[i]] = A.shape[i];
        }
    }

    for (int i = 0; i < B_perm.size(); i++) {
        if (out[B_perm[i]] == 0) {
            out[B_perm[i]] = B.shape[i];
        }
    }

    return out;
}

std::vector<size_t> einsums::tensor_algebra::detail::get_dim_ranges_for_many(pybind11::buffer_info const &A, std::vector<int> const &A_perm,
                                                                             pybind11::buffer_info const &B, std::vector<int> const &B_perm,
                                                                             int unique_indices) {
    std::vector<size_t> out(unique_indices);
    for (int i = 0; i < unique_indices; i++) {
        out[i] = 0;
    }

    for (int i = 0; i < A_perm.size(); i++) {
        if (out[A_perm[i]] == 0) {
            out[A_perm[i]] = A.shape[i];
        }
    }

    for (int i = 0; i < B_perm.size(); i++) {
        if (out[B_perm[i]] == 0) {
            out[B_perm[i]] = B.shape[i];
        }
    }

    return out;
}

void PyEinsumGenericPlan::execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                                  pybind11::buffer const &A, pybind11::buffer const &B) const {
    pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
    if (C_info.format != A_info.format || C_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C_info.format == pybind11::format_descriptor<float>::format()) {
        execute_generic<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_generic<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_generic<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<double>::format()) {
        execute_generic<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}

#ifdef EINSUMS_COMPUTE_CODE
void PyEinsumGenericPlan::execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                                  python::PyGPUView const &A, python::PyGPUView const &B) const {
    if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
        execute_generic_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_generic_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A,
                                                  B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_generic_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A,
                                                 B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
        execute_generic_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}
#endif

void PyEinsumDotPlan::execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                              pybind11::buffer const &A, pybind11::buffer const &B) const {
    pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
    if (C_info.format != A_info.format || C_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C_info.format == pybind11::format_descriptor<float>::format()) {
        execute_imp<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<double>::format()) {
        execute_imp<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}

#ifdef EINSUMS_COMPUTE_CODE
void PyEinsumDotPlan::execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                              python::PyGPUView const &A, python::PyGPUView const &B) const {
    if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
        execute_imp_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
        execute_imp_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}
#endif

void PyEinsumDirectProductPlan::execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                                        pybind11::buffer const &A, pybind11::buffer const &B) const {
    pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
    if (C_info.format != A_info.format || C_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C_info.format == pybind11::format_descriptor<float>::format()) {
        execute_imp<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<double>::format()) {
        execute_imp<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}

#ifdef EINSUMS_COMPUTE_CODE
void PyEinsumDirectProductPlan::execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                                        python::PyGPUView const &A, python::PyGPUView const &B) const {
    if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
        execute_imp_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
        execute_imp_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}
#endif

void PyEinsumGerPlan::execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                              pybind11::buffer const &A, pybind11::buffer const &B) const {
    pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
    // Check to make sure that we can use the library function.
    size_t C_check = C_info.itemsize, A_check = A_info.itemsize, B_check = B_info.itemsize;

    bool use_generic = false;

    for (int i = C_info.ndim - 1; i >= 0; i--) {
        if (C_check != C_info.strides[i]) {
            use_generic = true;
            break;
        }
        C_check *= C_info.shape[i];
    }

    for (int i = A_info.ndim - 1; i >= 0; i--) {
        if (A_check != A_info.strides[i]) {
            use_generic = true;
            break;
        }
        A_check *= A_info.shape[i];
    }

    for (int i = B_info.ndim - 1; i >= 0; i--) {
        if (B_check != B_info.strides[i]) {
            use_generic = true;
            break;
        }
        B_check *= B_info.shape[i];
    }

    if (use_generic) {
        PyEinsumGenericPlan::execute(C_prefactor, C, AB_prefactor, A, B);
        return;
    }
    if (C_info.format != A_info.format || C_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C_info.format == pybind11::format_descriptor<float>::format()) {
        execute_imp<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<double>::format()) {
        execute_imp<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}

#ifdef EINSUMS_COMPUTE_CODE
void PyEinsumGerPlan::execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                              python::PyGPUView const &A, python::PyGPUView const &B) const {
    // Check to make sure that we can use the library function.
    size_t C_check = C.itemsize(), A_check = A.itemsize(), B_check = B.itemsize();

    bool use_generic = false;

    for (int i = C.rank() - 1; i >= 0; i--) {
        if (C_check != C.stride(i)) {
            use_generic = true;
            break;
        }
        C_check *= C.dim(i);
    }

    for (int i = A.rank() - 1; i >= 0; i--) {
        if (A_check != A.stride(i)) {
            use_generic = true;
            break;
        }
        A_check *= A.dim(i);
    }

    for (int i = B.rank() - 1; i >= 0; i--) {
        if (B_check != B.stride(i)) {
            use_generic = true;
            break;
        }
        B_check *= B.dim(i);
    }

    if (use_generic) {
        PyEinsumGenericPlan::execute(C_prefactor, C, AB_prefactor, A, B);
        return;
    }
    if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
        execute_imp_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
        execute_imp_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}
#endif

void PyEinsumGemvPlan::execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                               pybind11::buffer const &A, pybind11::buffer const &B) const {
    pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
    // Check to make sure that we can use the library function.
    size_t C_check = C_info.itemsize, A_check = A_info.itemsize, B_check = B_info.itemsize;

    bool use_generic = false;

    for (int i = C_info.ndim - 1; i >= 0; i--) {
        if (C_check != C_info.strides[i]) {
            use_generic = true;
            break;
        }
        C_check *= C_info.shape[i];
    }

    for (int i = A_info.ndim - 1; i >= 0; i--) {
        if (A_check != A_info.strides[i]) {
            use_generic = true;
            break;
        }
        A_check *= A_info.shape[i];
    }

    for (int i = B_info.ndim - 1; i >= 0; i--) {
        if (B_check != B_info.strides[i]) {
            use_generic = true;
            break;
        }
        B_check *= B_info.shape[i];
    }

    if (use_generic) {
        PyEinsumGenericPlan::execute(C_prefactor, C, AB_prefactor, A, B);
        return;
    }
    if (C_info.format != A_info.format || C_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C_info.format == pybind11::format_descriptor<float>::format()) {
        execute_imp<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<double>::format()) {
        execute_imp<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}

#ifdef EINSUMS_COMPUTE_CODE
void PyEinsumGemvPlan::execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                               python::PyGPUView const &A, python::PyGPUView const &B) const {
    // Check to make sure that we can use the library function.
    size_t C_check = C.itemsize(), A_check = A.itemsize(), B_check = B.itemsize();

    bool use_generic = false;

    for (int i = C.rank() - 1; i >= 0; i--) {
        if (C_check != C.stride(i)) {
            use_generic = true;
            break;
        }
        C_check *= C.dim(i);
    }

    for (int i = A.rank() - 1; i >= 0; i--) {
        if (A_check != A.stride(i)) {
            use_generic = true;
            break;
        }
        A_check *= A.dim(i);
    }

    for (int i = B.rank() - 1; i >= 0; i--) {
        if (B_check != B.stride(i)) {
            use_generic = true;
            break;
        }
        B_check *= B.dim(i);
    }

    if (use_generic) {
        PyEinsumGenericPlan::execute(C_prefactor, C, AB_prefactor, A, B);
        return;
    }
    if (!_swap_AB) {
        if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
        } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
            execute_imp_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
        } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
            execute_imp_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A,
                                                  B);
        } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
            execute_imp_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A,
                                                 B);
        } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
            execute_imp_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
        } else {
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
        }
    } else {
        if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
        } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
            execute_imp_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), B, A);
        } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
            execute_imp_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), B,
                                                  A);
        } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
            execute_imp_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), B,
                                                 A);
        } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
            execute_imp_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), B, A);
        } else {
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
        }
    }
}
#endif

void PyEinsumGemmPlan::execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                               pybind11::buffer const &A, pybind11::buffer const &B) const {
    pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
    // Check to make sure that we can use the library function.
    size_t C_check = C_info.itemsize, A_check = A_info.itemsize, B_check = B_info.itemsize;

    bool use_generic = false;

    for (int i = C_info.ndim - 1; i >= 0; i--) {
        if (C_check != C_info.strides[i]) {
            use_generic = true;
            break;
        }
        C_check *= C_info.shape[i];
    }

    for (int i = A_info.ndim - 1; i >= 0; i--) {
        if (A_check != A_info.strides[i]) {
            use_generic = true;
            break;
        }
        A_check *= A_info.shape[i];
    }

    for (int i = B_info.ndim - 1; i >= 0; i--) {
        if (B_check != B_info.strides[i]) {
            use_generic = true;
            break;
        }
        B_check *= B_info.shape[i];
    }

    if (use_generic) {
        PyEinsumGenericPlan::execute(C_prefactor, C, AB_prefactor, A, B);
        return;
    }
    if (C_info.format != A_info.format || C_info.format != B_info.format) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C_info.format == pybind11::format_descriptor<float>::format()) {
        execute_imp<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C_info.format == pybind11::format_descriptor<double>::format()) {
        execute_imp<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}

#ifdef EINSUMS_COMPUTE_CODE
void PyEinsumGemmPlan::execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                               python::PyGPUView const &A, python::PyGPUView const &B) const {
    // Check to make sure that we can use the library function.
    size_t C_check = C.itemsize(), A_check = A.itemsize(), B_check = B.itemsize();

    bool use_generic = false;

    for (int i = C.rank() - 1; i >= 0; i--) {
        if (C_check != C.stride(i)) {
            use_generic = true;
            break;
        }
        C_check *= C.dim(i);
    }

    for (int i = A.rank() - 1; i >= 0; i--) {
        if (A_check != A.stride(i)) {
            use_generic = true;
            break;
        }
        A_check *= A.dim(i);
    }

    for (int i = B.rank() - 1; i >= 0; i--) {
        if (B_check != B.stride(i)) {
            use_generic = true;
            break;
        }
        B_check *= B.dim(i);
    }

    if (use_generic) {
        PyEinsumGenericPlan::execute(C_prefactor, C, AB_prefactor, A, B);
        return;
    }
    if (C.fmt_spec() != A.fmt_spec() || C.fmt_spec() != B.fmt_spec()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not handle tensors with different dtypes (yet)!");
    } else if (C.fmt_spec() == pybind11::format_descriptor<float>::format()) {
        execute_imp_gpu<float>(C_prefactor.cast<float>(), C, AB_prefactor.cast<float>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<double>>::format()) {
        execute_imp_gpu<std::complex<double>>(C_prefactor.cast<std::complex<double>>(), C, AB_prefactor.cast<std::complex<double>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<std::complex<float>>::format()) {
        execute_imp_gpu<std::complex<float>>(C_prefactor.cast<std::complex<float>>(), C, AB_prefactor.cast<std::complex<float>>(), A, B);
    } else if (C.fmt_spec() == pybind11::format_descriptor<double>::format()) {
        execute_imp_gpu<double>(C_prefactor.cast<double>(), C, AB_prefactor.cast<double>(), A, B);
    } else {
        EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not handle tensor type!");
    }
}
#endif
