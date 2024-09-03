#include <pybind11/pybind11.h>

#include "einsums.hpp"

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::python;

using namespace py = pybind11;

PyEinsumPlan::PyEinsumPlan(python::detail::PyPlanDataType C_datatype, python::detail::PyPlanDataType A_datatype,
                           python::detail::PyPlanDataType B_datatype, python::detail::PyPlanUnit unit)
    : _C_datatype{C_datatype}, _A_datatype{A_datatype}, _B_datatype{B_datatype}, _unit{unit} {
}

PyEinsumGenericPlan::PyEinsumGenericPlan(std::vector<int> C_permute, std::vector<int> A_permute, std::vector<int> B_permute,
                                         const PyEinsumPlan &plan_base)
    : PyEinsumPlan(plan_base), _C_permute{C_permute}, _A_permute{A_permute}, _B_permute{B_permute} {
}

PyEinsumDotPlan::PyEinsumDotPlan(const PyEinsumPlan &plan_base) : PyEinsumPlan(plan_base) {
}

PyEinsumDirectProductPlan::PyEinsumDirectProductPlan(const PyEinsumPlan &plan_base) : PyEinsumPlan(plan_base) {
}

PyEinsumGerPlan::PyEinsumGerPlan(const PyEinsumPlan &plan_base) : PyEinsumPlan(plan_base) {
}

PyEinsumGemvPlan::PyEinsumGemvPlan(const PyEinsumPlan &plan_base) : PyEinsumPlan(plan_base) {
}

PyEinsumGemmPlan::PyEinsumGemmPlan(const PyEinsumPlan &plan_base) : PyEinsumPlan(plan_base) {
}

einsums::tensor_algebra::compile_plan(std::string C_indices, std::string A_indices, std::string B_indices,
                                      python::detail::PyPlanUnit unit = python::detail::CPU) {

    size_t ARank = A_indices.size();
    size_t BRank = B_indices.size();
    size_t CRank = C_indices.size();

    std::string linksAB = "";

    for (int i = 0; i < ARank; i++) {
        for (int j = 0; j < BRank; j++) {
            if (A_indices[i] == B_indices[j]) {
                linksAB += A_indices[i];
            }
        }
    }

    std::string links = "";

    for (int i = 0; i < linksAB.size(); i++) {
        bool add = true;
        for (int j = 0; j < CRank; j++) {
            if (linksAB[i] == C_indices[j]) {
                add = false;
                break;
            }
        }
        if (add) {
            links += linksAB[i];
        }
    }

    std::string CAlinks = "";

    for (int i = 0; i < CRank; i++) {
        for (int j = 0; j < ARank; j++) {
            if (C_indices[i] == A_indices[j]) {
                CAlinks += C_indices[i];
            }
        }
    }

    std::string CBlinks = "";

    for (int i = 0; i < CRank; i++) {
        for (int j = 0; j < BRank; j++) {
            if (C_indices[i] == B_indices[j]) {
                CBlinks += C_indices[i];
            }
        }
    }

    std::string CminusA = "";

    for (int i = 0; i < CRank; i++) {
        bool add = true;
        for (int j = 0; j < ARank; j++) {
            if (C_indices[i] == A_indices[j]) {
                add = false;
                break;
            }
        }
        if (add) {
            CminusA += C_indices[i];
        }
    }

    std::string CminusB = "";

    for (int i = 0; i < CRank; i++) {
        bool add = true;
        for (int j = 0; j < BRank; j++) {
            if (C_indices[i] == B_indices[j]) {
                add = false;
                break;
            }
        }
        if (add) {
            CminusB += C_indices[i];
        }
    }

    bool have_remaining_indices_in_CminusA = CminusA.size() > 0;
    bool have_remaining_indices_in_CminusB = CminusB.size() > 0;

    std::string A_only = "";

    for (int i = 0; i < ARank; i++) {
        bool add = true;
        for (int j = 0; j < links.size(); j++) {
            if (A_indices[i] == links[j]) {
                add = false;
                break;
            }
        }
        if (add) {
            A_only += A_indices[i];
        }
    }

    std::string B_only = "";

    for (int i = 0; i < BRank; i++) {
        bool add = true;
        for (int j = 0; j < links.size(); j++) {
            if (B_indices[i] == links[j]) {
                add = false;
                break;
            }
        }
        if (add) {
            B_only += B_indices[i];
        }
    }

    std::string A_unique = "";

    for (int i = 0; i < ARank; i++) {
        bool add = true;
        for (int j = 0; j < A_unique.size(); j++) {
            if (A_unique[j] == A_indices[i]) {
                add = false;
                break;
            }
        }
        if (add) {
            A_unique += A_indices[i];
        }
    }

    std::string B_unique = "";

    for (int i = 0; i < BRank; i++) {
        bool add = true;
        for (int j = 0; j < B_unique.size(); j++) {
            if (B_unique[j] == B_indices[i]) {
                add = false;
                break;
            }
        }
        if (add) {
            B_unique += B_indices[i];
        }
    }

    std::string C_unique = "";

    for (int i = 0; i < CRank; i++) {
        bool add = true;
        for (int j = 0; j < C_unique.size(); j++) {
            if (C_unique[j] == C_indices[i]) {
                add = false;
                break;
            }
        }
        if (add) {
            C_unique += C_indices[i];
        }
    }

    std::string link_unique = "";

    for (int i = 0; i < links.size(); i++) {
        bool add = true;
        for (int j = 0; j < link_unique.size(); j++) {
            if (link_unique[j] == links[i]) {
                add = false;
                break;
            }
        }
        if (add) {
            link_unique += link_indices[i];
        }
    }

    bool A_hadamard_found = ARank != A_unique.size();
    bool B_hadamard_found = BRank != B_unique.size();
    bool C_hadamard_found = CRank != C_unique.size();

    
}