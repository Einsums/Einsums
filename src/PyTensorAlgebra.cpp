#include "include/einsums/Python.hpp"
#include "include/einsums/python/PyTensorAlgebra.hpp"

#include <pybind11/pybind11.h>

#include "einsums.hpp"

using namespace std;
using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::python;

namespace py = pybind11;

PyEinsumGenericPlan::PyEinsumGenericPlan(int num_inds, std::vector<int> C_permute, std::vector<int> A_permute, std::vector<int> B_permute,
                                         einsums::python::detail::PyPlanUnit unit)
    : _num_inds{num_inds}, _C_permute{C_permute}, _A_permute{A_permute}, _B_permute{B_permute}, _unit{unit} {
}

PyEinsumDotPlan::PyEinsumDotPlan(const PyEinsumGenericPlan &plan_base) : PyEinsumGenericPlan(plan_base) {
}

PyEinsumDirectProductPlan::PyEinsumDirectProductPlan(const PyEinsumGenericPlan &plan_base) : PyEinsumGenericPlan(plan_base) {
}

PyEinsumGerPlan::PyEinsumGerPlan(const PyEinsumGenericPlan &plan_base) : PyEinsumGenericPlan(plan_base) {
}

PyEinsumGemvPlan::PyEinsumGemvPlan(const PyEinsumGenericPlan &plan_base) : PyEinsumGenericPlan(plan_base) {
}

PyEinsumGemmPlan::PyEinsumGemmPlan(const PyEinsumGenericPlan &plan_base) : PyEinsumGenericPlan(plan_base) {
}

string einsums::tensor_algebra::detail::intersect(const string &st1, const string &st2) {
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

static string difference(const string &st1, const string &st2) {
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

static string unique(const string &str) {
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

static vector<pair<char, size_t>> find_ind_with_pos(const string &st1, const string &st2) {
    vector<pair<char, size_t>> out;

    for (int i = 0; i < st1.length(); i++) {
        for (int j = 0; j < st2.length(); j++) {
            if (st1[i] == st2[j]) {
                out.push_back(pair<char, size_t>(st1[i], j));
            }
        }
    }

    return out;
}

static bool contiguous_indices(const vector<pair<char, size_t>> &pairs) {
    for (int i = 0; i < pairs.size() - 1; i++) {
        if (pairs[i].second != pairs[i + 1].second - 1) {
            return false;
        }
    }
    return true;
}

static bool same_ordering(const vector<pair<char, size_t>> &x, const vector<pair<char, size_t>> &y) {
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

static vector<int> create_perm_table(const string &indices, const string &unique_indices) {
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

std::shared_ptr<einsums::tensor_algebra::PyEinsumGenericPlan> einsums::tensor_algebra::compile_plan(std::string                C_indices,
                                                                                                    std::string                A_indices,
                                                                                                    std::string                B_indices,
                                                                                                    python::detail::PyPlanUnit unit) {
    using namespace einsums::tensor_algebra::detail;
    size_t ARank = A_indices.size();
    size_t BRank = B_indices.size();
    size_t CRank = C_indices.size();

    std::string linksAB = intersect(A_indices, B_indices);

    std::string links = difference(linksAB, C_indices);

    std::string CAlinks = intersect(C_indices, A_indices);

    std::string CBlinks = intersect(C_indices, B_indices);

    std::string CminusA = difference(C_indices, A_indices);

    std::string CminusB = difference(C_indices, A_indices);

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
    bool same_ordering_target_position_in_CA = same_ordering(target_position_in_A, target_position_in_C);
    bool same_ordering_target_position_in_CB = same_ordering(target_position_in_B, target_position_in_C);

    bool C_exactly_matches_A = (A_indices == C_indices);
    bool C_exactly_matches_B = (B_indices == C_indices);
    bool A_exactly_matches_B = (A_indices == B_indices);

    bool is_gemm_possible = have_remaining_indices_in_CminusA && have_remaining_indices_in_CminusB && contiguous_link_position_in_A &&
                            contiguous_link_position_in_B && contiguous_target_position_in_A && contiguous_target_position_in_B &&
                            contiguous_A_targets_in_C && contiguous_B_targets_in_C && same_ordering_link_position_in_AB &&
                            same_ordering_target_position_in_CA && same_ordering_target_position_in_CB && !A_hadamard_found &&
                            !B_hadamard_found && !C_hadamard_found;
    bool is_gemv_possible = contiguous_link_position_in_A && contiguous_link_position_in_B && contiguous_target_position_in_A &&
                            same_ordering_link_position_in_AB && same_ordering_target_position_in_CA &&
                            !same_ordering_target_position_in_CB && B_target_position_in_C.size() == 0 && !A_hadamard_found &&
                            !B_hadamard_found && !C_hadamard_found;
    bool element_wise_multiplication =
        C_exactly_matches_A && C_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    bool dot_product   = C_indices.length() == 0 && A_exactly_matches_B && !A_hadamard_found && !B_hadamard_found && !C_hadamard_found;
    bool outer_product = linksAB.size() == 0 && contiguous_target_position_in_A && contiguous_target_position_in_B && !A_hadamard_found &&
                         !B_hadamard_found && !C_hadamard_found;

    string unique_indices = unique(A_indices + B_indices + C_indices);

    vector<int> A_perm = create_perm_table(A_indices, unique_indices), B_perm = create_perm_table(B_indices, unique_indices),
                C_perm = create_perm_table(C_indices, unique_indices);

    PyEinsumGenericPlan base(unique_indices.length(), C_perm, A_perm, B_perm, unit);

    if (dot_product) {
        return make_shared<PyEinsumDotPlan>(base);
    } else if (element_wise_multiplication) {
        return make_shared<PyEinsumDirectProductPlan>(base);
    } else if (outer_product) {
        return make_shared<PyEinsumGerPlan>(base);
    } else if (is_gemv_possible) {
        return make_shared<PyEinsumGemvPlan>(base);
    } else if (is_gemm_possible) {
        return make_shared<PyEinsumGemmPlan>(base);
    } else {
        return make_shared<PyEinsumGenericPlan>(base);
    }
}

std::vector<size_t> einsums::tensor_algebra::detail::get_dim_ranges_for_many(const pybind11::array &C, const std::vector<int> &C_perm,
                                                                             const pybind11::array &A, const std::vector<int> &A_perm,
                                                                             const pybind11::array &B, const std::vector<int> &B_perm,
                                                                             int unique_indices) {
    std::vector<size_t> out(unique_indices);
    for (int i = 0; i < unique_indices; i++) {
        out[i] = 0;
    }

    for (int i = 0; i < C_perm.size(); i++) {
        if (out[C_perm[i]] == 0) {
            out[C_perm[i]] = C.shape(i);
        }
    }

    for (int i = 0; i < A_perm.size(); i++) {
        if (out[A_perm[i]] == 0) {
            out[A_perm[i]] = A.shape(i);
        }
    }

    for (int i = 0; i < B_perm.size(); i++) {
        if (out[B_perm[i]] == 0) {
            out[B_perm[i]] = B.shape(i);
        }
    }

    return out;
}

void PyEinsumGenericPlan::execute(float C_prefactor, pybind11::array_t<float> &C, float AB_prefactor, const pybind11::array_t<float> &A,
                                  const pybind11::array_t<float> &B) const {
    execute_imp<float>(C_prefactor, C, AB_prefactor, A, B);
}

void PyEinsumGenericPlan::execute(double C_prefactor, pybind11::array_t<double> &C, double AB_prefactor, const pybind11::array_t<double> &A,
                                  const pybind11::array_t<double> &B) const {
    execute_imp<double>(C_prefactor, C, AB_prefactor, A, B);
}
