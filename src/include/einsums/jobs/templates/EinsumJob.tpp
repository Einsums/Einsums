/**
 * @file EinsumJob.tpp
 *
 * Contains definitions for the EinsumJob class
 */

#pragma once

#include <atomic>
#include <type_traits>

#include "einsums/jobs/EinsumJob.hpp"

#include "einsums/TensorAlgebra.hpp"
#include "einsums/_Common.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @def template_def
 *
 * The template definition for the EinsumJob class is long. This macro replaces the template line so that writing code is not tedious.
 * Deleted at the end of the file. Not valid outside this file.
 */
#define template_def                                                                                                                       \
    template <typename AType, typename ABDataType, typename BType, typename CType, typename CDataType, typename CIndices,                  \
              typename AIndices, typename BIndices>

/**
 * @def einsum_job
 *
 * The template specialization of EinsumJob is long. This macro replaces the class name and template parameters to make coding easier.
 * Deleted at the end of the file. Not valid outside this file.
 */
#define einsum_job EinsumJob<AType, ABDataType, BType, CType, CDataType, CIndices, AIndices, BIndices>

template_def
einsum_job::EinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WritePromise<CType>> C, const ABDataType AB_prefactor,
              const AIndices &As, std::shared_ptr<ReadPromise<AType>> A, const BIndices &Bs, std::shared_ptr<ReadPromise<BType>> B)
        : Job(), _C_prefactor(C_prefactor), _AB_prefactor(AB_prefactor), _Cs(Cs), _As(As), _Bs(Bs), _running(false), _done(false) {
        _A = A;
        _B = B;
        _C = C;
    }

template_def
void einsum_job::run() {
        std::atomic_thread_fence(std::memory_order_acq_rel);
        _running = true;
        auto A   = _A->get();
        auto B   = _B->get();
        auto C   = _C->get();

        einsums::tensor_algebra::einsum(_C_prefactor, _Cs, C.get(), _AB_prefactor, _As, *A, _Bs, *B);

        _C->release();
        _A->release();
        _B->release();
        _running = false;
        _done    = true;
        std::atomic_thread_fence(std::memory_order_acq_rel);
    }

template_def
bool einsum_job::is_runnable() const { return _A->ready() && _B->ready() && _C->ready() && !_running && !_done; }

template_def
bool einsum_job::is_running() const { return this->_running; }


template_def
bool einsum_job::is_finished() const { return this->_done; }


#undef einsum_job
#undef template_def
END_EINSUMS_NAMESPACE_HPP(einsums::jobs)