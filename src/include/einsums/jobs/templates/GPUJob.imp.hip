/**
 * @file EinsumJob.tpp
 *
 * Contains definitions for the EinsumJob class
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/GPUTensorAlgebra.hpp"
#include "einsums/jobs/GPUJob.hpp"
#include "einsums/jobs/GPUJob.hpp"
#include "einsums/jobs/ThreadPool.hpp"

#include <atomic>
#include <thread>
#include <type_traits>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs::gpu)

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
#define einsum_job GPUEinsumJob<AType, ABDataType, BType, CType, CDataType, CIndices, AIndices, BIndices>

template_def einsum_job::GPUEinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WritePromise<CType>> C,
                                   const ABDataType AB_prefactor, const AIndices &As, std::shared_ptr<ReadPromise<AType>> A,
                                   const BIndices &Bs, std::shared_ptr<ReadPromise<BType>> B, int num_threads_param, bool is_limit_hard)
    : Job(), _C_prefactor(C_prefactor), _AB_prefactor(AB_prefactor), _Cs(Cs), _As(As), _Bs(Bs), __num_threads(num_threads_param),
      __hard_limit(is_limit_hard) {
    _A    = A;
    _B    = B;
    _C    = C;
    work  = new ABDataType[__num_threads];
    synch = 0;

    __default_stream = true;
    __stream = 0;
}

template_def einsum_job::GPUEinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WritePromise<CType>> C,
                                   const ABDataType AB_prefactor, const AIndices &As, std::shared_ptr<ReadPromise<AType>> A,
                                   const BIndices &Bs, std::shared_ptr<ReadPromise<BType>> B, hipStream_t stream, int num_threads_param, bool is_limit_hard)
    : Job(), _C_prefactor(C_prefactor), _AB_prefactor(AB_prefactor), _Cs(Cs), _As(As), _Bs(Bs), __num_threads(num_threads_param),
      __hard_limit(is_limit_hard) {
    _A    = A;
    _B    = B;
    _C    = C;
    work  = new ABDataType[__num_threads];
    synch = 0;

    __default_stream = false;
    __stream = stream;
}

template_def einsum_job::~GPUEinsumJob() {
    delete[] work;
}

template_def void einsum_job::run() {
    std::atomic_thread_fence(std::memory_order_acq_rel);
    auto A = _A->get();
    auto B = _B->get();
    auto C = _C->get();

    this->set_state(einsums::jobs::detail::RUNNING);

    if(this->__default_stream) {
        einsums::tensor_algebra::gpu::einsum(_C_prefactor, _Cs, &C, _AB_prefactor, _As, A, _Bs, B);

        einsums::gpu::hip_catch(hipDeviceSynchronize());
    } else {
        einsums::tensor_algebra::gpu::einsum(_C_prefactor, _Cs, &C, _AB_prefactor, _As, A, _Bs, B, __stream);
        hipEvent_t wait_event;

        einsums::gpu::hip_catch(hipEventCreate(&wait_event));

        einsums::gpu::hip_catch(hipEventRecord(wait_event, __stream));

        while(hipEventQuery(wait_event) == hipErrorNotReady) {
            std::this_thread::yield();
        }
    }


    _C->release();
    _A->release();
    _B->release();

    this->set_state(einsums::jobs::detail::FINISHED);

    std::atomic_thread_fence(std::memory_order_acq_rel);
}

#undef einsum_job
#undef template_def
END_EINSUMS_NAMESPACE_HPP(einsums::jobs)