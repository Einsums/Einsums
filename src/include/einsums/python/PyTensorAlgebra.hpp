#pragma once

#include "einsums/python/Python.hpp"
#include "einsums/utility/TensorBases.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace einsums::tensor_algebra {

/**
 * @class PyEinsumPlan
 *
 * @brief Holds data for performing an einsum calculation in Python.
 *
 * Using these plans, we should be able to simulate the idea behind the compile-time optimizations.
 */
class PyEinsumPlan {
  protected:
    python::detail::PyPlanDataType _C_datatype, _A_datatype, _B_datatype;
    python::detail::PyPlanUnit     _unit;

  public:
    PyEinsumPlan() = delete;

    PyEinsumPlan(python::detail::PyPlanDataType C_datatype, python::detail::PyPlanDataType A_datatype,
                 python::detail::PyPlanDataType B_datatype, python::detail::PyPlanUnit unit);

    PyEinsumPlan(const PyEinsumPlan &) = default;

    virtual ~PyEinsumPlan() = default;
};

/**
 * @class PyEinsumGenericPlan
 *
 * @brief Holds the info for the generic algorithm, called when all the optimizations fail.
 */
class PyEinsumGenericPlan : public PyEinsumPlan {
  private:
    std::vector<int> _C_permute, _A_permute, _B_permute;

  public:
    PyEinsumGenericPlan() = delete;
    PyEinsumGenericPlan(std::vector<int> C_permute, std::vector<int> A_permute, std::vector<int> B_permute, const PyEinsumPlan &plan_base);
    PyEinsumGenericPlan(const PyEinsumGenericPlan &) = default;
    ~PyEinsumGenericPlan()                           = default;

    template <typename CDataType, typename ABDataType>
    void execute(CDataType C_prefactor, tensor_props::PyTensorBase *C, ABDataType AB_prefactor, const tensor_props::PyTensorBase *A,
                 const tensor_props::PyTensorBase *B) const {}
};

class PyEinsumDotPlan : public PyEinsumPlan {
  public:
    PyEinsumDotPlan() = delete;
    PyEinsumDotPlan(const PyEinsumPlan &plan_base);
    PyEinsumDotPlan(const PyEinsumDotPlan &) = default;
    ~PyEinsumDotPlan()                       = default;

    template <typename CDataType, typename ABDataType>
    void execute(CDataType C_prefactor, tensor_props::PyTensorBase *C, ABDataType AB_prefactor, const tensor_props::PyTensorBase *A,
                 const tensor_props::PyTensorBase *B) const {}
};

class PyEinsumDirectProductPlan : public PyEinsumPlan {
  public:
    PyEinsumDirectProductPlan() = delete;
    PyEinsumDirectProductPlan(const PyEinsumPlan &plan_base);
    PyEinsumDirectProductPlan(const PyEinsumDirectProductPlan &) = default;
    ~PyEinsumDirectProductPlan()                                 = default;

    template <typename CDataType, typename ABDataType>
    void execute(CDataType C_prefactor, tensor_props::PyTensorBase *C, ABDataType AB_prefactor, const tensor_props::PyTensorBase *A,
                 const tensor_props::PyTensorBase *B) const {}
};

class PyEinsumGerPlan : public PyEinsumPlan {
  public:
    PyEinsumGerPlan() = delete;
    PyEinsumGerPlan(const PyEinsumPlan &plan_base);
    PyEinsumGerPlan(const PyEinsumGerPlan &) = default;
    ~PyEinsumGerPlan()                       = default;

    template <typename CDataType, typename ABDataType>
    void execute(CDataType C_prefactor, tensor_props::PyTensorBase *C, ABDataType AB_prefactor, const tensor_props::PyTensorBase *A,
                 const tensor_props::PyTensorBase *B) const {}
};

class PyEinsumGemvPlan : public PyEinsumPlan {
  public:
    PyEinsumGemvPlan() = delete;
    PyEinsumGemvPlan(const PyEinsumPlan &plan_base);
    PyEinsumGemvPlan(const PyEinsumGemvPlan &) = default;
    ~PyEinsumGemvPlan()                        = default;

    template <typename CDataType, typename ABDataType>
    void execute(CDataType C_prefactor, tensor_props::PyTensorBase *C, ABDataType AB_prefactor, const tensor_props::PyTensorBase *A,
                 const tensor_props::PyTensorBase *B) const {

    };
};

class PyEinsumGemmPlan : public PyEinsumPlan {
  public:
    PyEinsumGemmPlan() = delete;
    PyEinsumGemmPlan(const PyEinsumPlan &plan_base);
    PyEinsumGemmPlan(const PyEinsumGemmPlan &) = default;
    ~PyEinsumGemmPlan()                        = default;

    template <typename CDataType, typename ABDataType>
    void execute(CDataType C_prefactor, tensor_props::PyTensorBase *C, ABDataType AB_prefactor, const tensor_props::PyTensorBase *A,
                 const tensor_props::PyTensorBase *B) const {}
};

EINSUMS_EXPORT PyEinsumPlan compile_plan(std::string C_indices, std::string A_indices, std::string B_indices,
                                         python::detail::PyPlanUnit unit = python::detail::CPU);

} // namespace einsums::tensor_algebra
