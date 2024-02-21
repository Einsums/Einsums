#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/Job.hpp"
#include "einsums/jobs/ReadPromise.hpp"
#include "einsums/jobs/WritePromise.hpp"

#include <atomic>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs::gpu)

/**
 * @struct EinsumJob
 *
 * Holds information for running einsum as a job.
 */
template <typename AType, typename ABDataType, typename BType, typename CType, typename CDataType, typename CIndices, typename AIndices,
          typename BIndices>
class GPUEinsumJob : public Job {
  protected:
    /**
     * @var _A
     *
     * A read promise to the left tensor in the calculation.
     */
    std::shared_ptr<ReadPromise<AType>> _A;

    /**
     * @var _B
     *
     * A read promise to the right tensor in the calculation.
     */
    std::shared_ptr<ReadPromise<BType>> _B;

    /**
     * @var _C
     *
     * A write promise to the result tensor.
     */
    std::shared_ptr<WritePromise<CType>> _C;

    /**
     * @var _C_prefactor
     *
     * The prefactor for the result tensor.
     */
    const CDataType _C_prefactor;

    /**
     * @var _AB_prefactor
     *
     * The prefactor to multiply the A and B terms by.
     */
    const ABDataType _AB_prefactor;

    /**
     * @var _Cs
     *
     * The indices for the result tensor.
     */
    const CIndices &_Cs;

    /**
     * @var _As
     *
     * The indices for the left tensor.
     */
    const AIndices &_As;

    /**
     * @var _Bs
     *
     * The indices for the right tensor.
     */
    const BIndices &_Bs;

    /**
     * @var work
     *
     * A work array.
     */
    ABDataType *work;

    /**
     * Synchronization value.
     */
    std::atomic_int synch;

    /**
     * @var num_threads
     *
     * The number of threads that the job requests.
     */
    int __num_threads;

    /**
     * @var hard_limit
     *
     * True if the job can only run on num_threads threads. False if the job can run on fewer.
     */
    bool __hard_limit;

    /**
     * @var default_stream
     *
     * True if the job should be placed on the default stream. False if a stream has been specified.
     */
    bool __default_stream

    /**
     * @var stream
     *
     * The stream to run the job. Ignored if __default_stream is true.
     */
    hipStream_t stream;

  public:
    /**
     * Constructor.
     */
    GPUEinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WritePromise<CType>> C, const ABDataType AB_prefactor,
              const AIndices &As, std::shared_ptr<ReadPromise<AType>> A, const BIndices &Bs, std::shared_ptr<ReadPromise<BType>> B,
              int num_threads_param = 1, bool is_limit_hard = true);
    GPUEinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WritePromise<CType>> C, const ABDataType AB_prefactor,
              const AIndices &As, std::shared_ptr<ReadPromise<AType>> A, const BIndices &Bs, std::shared_ptr<ReadPromise<BType>> B,
              hipStream_t stream, int num_threads_param = 1, bool is_limit_hard = true);

    /**
     * Destructor.
     */
    virtual ~GPUEinsumJob();

    /*
     * Overrides for the base class.
     */

    /**
     * The function to run when the job is called.
     */
    void run() override;

    /**
     * Get number of threads requested.
     */
    int num_threads() override { return __num_threads; }

    /**
     * Check whether the number of threads is a hard limit, or if fewer can be requested.
     */
    bool can_have_fewer() override { return __hard_limit; }
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs::gpu)

#include "einsums/jobs/templates/GPUJob.imp.hip"