/**
 * @file Jobs.hpp
 *
 * Job queues and resource management.
 */

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

#include "einsums/TensorAlgebra.hpp"
#include "einsums/jobs/EinsumJob.hpp"
#include "einsums/jobs/Job.hpp"
#include "einsums/jobs/JobManager.hpp"
#include "einsums/jobs/ReadPromise.hpp"
#include "einsums/jobs/Resource.hpp"
#include "einsums/jobs/ThreadPool.hpp"
#include "einsums/jobs/Timeout.hpp"
#include "einsums/jobs/WritePromise.hpp"

#include <stdexcept>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

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
            std::shared_ptr<Resource<BType>> &B)
    -> std::shared_ptr<EinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>,
                                 std::tuple<BIndices...>>> & {
    // Start of function
    using outtype =
        EinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>;

    std::shared_ptr<WritePromise<CType>> C_lock = C->write_promise();
    std::shared_ptr<ReadPromise<AType>>  A_lock = A->read_promise();
    std::shared_ptr<ReadPromise<BType>>  B_lock = B->read_promise();
    std::shared_ptr<outtype>            *out =
        new std::shared_ptr<outtype>(std::make_shared<outtype>(C_prefactor, Cs, C_lock, AB_prefactor, As, A_lock, Bs, B_lock));

    // Queue the job.
    JobManager::get_singleton().queue_job(*out);

    return *out;
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
            std::shared_ptr<Resource<AType>> &A, const std::tuple<BIndices...> &B_indices, std::shared_ptr<Resource<BType>> &B)
    -> std::shared_ptr<
        EinsumJob<AType,
                  std::conditional_t<sizeof(typename AType::datatype) < sizeof(typename BType::datatype), typename BType::datatype,
                                     typename AType::datatype>,
                  BType, CType, typename CType::datatype, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>> & {
    return einsum((typename CType::datatype)0, C_indices, C,
                  (std::conditional_t<sizeof(typename AType::datatype) < sizeof(typename BType::datatype), typename BType::datatype,
                                      typename AType::datatype>)1,
                  A_indices, A, B_indices, B);
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)
