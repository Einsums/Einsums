#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

#include "einsums/TensorAlgebra.hpp"
#include "einsums/jobs/Job.hpp"
#include "einsums/jobs/GPUJob.hpp"
#include "einsums/jobs/JobManager.hpp"
#include "einsums/jobs/ReadPromise.hpp"
#include "einsums/jobs/Resource.hpp"
#include "einsums/jobs/ThreadPool.hpp"
#include "einsums/jobs/Timeout.hpp"
#include "einsums/jobs/WritePromise.hpp"

#include <stdexcept>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs::gpu)

/**
 * Creates an einsum job and adds it to the job queue.
 *
 * @param C_prefactor The prefactor for the C tensor.
 * @param Cs The C indices.
 * @param C The C tensor.
 * @param AB_prefactor The prefactor for A and B.
 * @param As The A indices.
 * @param A The A tensor.
 * @param Bs The B indices.
 * @param B The B tensor.
 * @return A shared pointer to the job created and queued by this function.
 */
template <typename AType, typename ABDataType, typename BType, typename CType, typename CDataType, typename... CIndices,
          typename... AIndices, typename... BIndices>
auto einsum(CDataType C_prefactor, const std::tuple<CIndices...> &Cs, std::shared_ptr<Resource<CType>> &C, const ABDataType AB_prefactor,
            const std::tuple<AIndices...> &As, std::shared_ptr<Resource<AType>> &A, const std::tuple<BIndices...> &Bs,
            std::shared_ptr<Resource<BType>> &B, hipStream_t stream = nullptr, int num_threads = 1, bool is_limit_hard = true)
    -> std::shared_ptr<GPUEinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>,
                                 std::tuple<BIndices...>>> {
    // Start of function
    using outtype =
        GPUEinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>;

    std::shared_ptr<WritePromise<CType>> C_lock = C->write_promise();
    std::shared_ptr<ReadPromise<AType>>  A_lock = A->read_promise();
    std::shared_ptr<ReadPromise<BType>>  B_lock = B->read_promise();
    outtype           *out =
        new outtype(C_prefactor, Cs, C_lock, AB_prefactor, As, A_lock, Bs, B_lock, stream, num_threads, is_limit_hard);

    // Queue the job.
    std::shared_ptr<outtype> out_weak = JobManager::get_singleton().queue_job(out);

    return out_weak;
}

/**
 * Creates an einsum job and adds it to the job queue. Has default prefactors.
 *
 * @param Cs The C indices.
 * @param C The C tensor.
 * @param As The A indices.
 * @param A The A tensor.
 * @param Bs The B indices.
 * @param B The B tensor.
 * @return A shared pointer to the job created and queued by this function.
 */
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, std::shared_ptr<Resource<CType>> &C, const std::tuple<AIndices...> &A_indices,
            std::shared_ptr<Resource<AType>> &A, const std::tuple<BIndices...> &B_indices, std::shared_ptr<Resource<BType>> &B,
            hipStream_t stream = nullptr, int num_threads = 1, bool is_limit_hard = true)
    -> std::shared_ptr<
        GPUEinsumJob<AType,
                  std::conditional_t<sizeof(typename AType::host_datatype) < sizeof(typename BType::host_datatype), typename BType::host_datatype,
                                     typename AType::host_datatype>,
                  BType, CType, typename CType::host_datatype, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>> {
    return einsum((typename CType::host_datatype)0, C_indices, C,
                  (std::conditional_t<sizeof(typename AType::host_datatype) < sizeof(typename BType::host_datatype), typename BType::host_datatype,
                                      typename AType::host_datatype>)1,
                  A_indices, A, B_indices, B, stream, num_threads, is_limit_hard);
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs::gpu)
