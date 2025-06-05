//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <source_location>

namespace einsums {
namespace gpu {

/**
 * @def get_worker_info
 * @brief Get the worker thread launch parameters on the GPU.
 */
#define get_worker_info(thread_id, num_threads)                                                                                            \
    num_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;                                                \
    thread_id   = threadIdx.x +                                                                                                            \
                blockDim.x * (threadIdx.y +                                                                                                \
                              blockDim.y * (threadIdx.z + blockDim.z * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))));

/**
 * @brief Get the internal hipBLAS handle.
 *
 * @return The current internal hipBLAS handle.
 */
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t get_blas_handle();
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t get_blas_handle(int thread_id);

/**
 * @brief Get the internal hipSolver handle.
 *
 * @return The current internal hipSolver handle.
 */
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t get_solver_handle();
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t get_solver_handle(int thread_id);

// Set the handles used internally. Useful to avoid creating multiple contexts.
/**
 * @brief Set the internal hipBLAS handle.
 *
 * @param value The new handle.
 *
 * @return The new handle.
 */
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value);
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value, int thread_id);

/**
 * @brief Set the internal hipSolver handle.
 *
 * @param value The new handle.
 *
 * @return The new handle.
 */
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t set_solver_handle(hipsolverHandle_t value);
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t set_solver_handle(hipsolverHandle_t value, int thread_id);

/**
 * Get an appropriate block size for a kernel.
 */
EINSUMS_HOST inline dim3 block_size(size_t compute_size) {
    if (compute_size < 32) {
        return dim3(32);
    } else if (compute_size < 256) {
        return dim3(compute_size & 0xf0);
    } else {
        return dim3(256);
    }
}

/**
 * Get an appropriate number of blocks for a kernel.
 */
EINSUMS_HOST inline dim3 blocks(size_t compute_size) {
    if (compute_size < 256) {
        return dim3(1);
    } else if (compute_size <= 4096) {
        return dim3(compute_size / 256);
    } else {
        return dim3(16);
    }
}

/**
 * @brief Wait on a stream.
 */
EINSUMS_HOST EINSUMS_EXPORT void stream_wait(hipStream_t stream);

/**
 * @brief Indicates that the next skippable wait on the current thread should be skipped.
 * Does not apply to stream_wait(stream) with the stream specified, all_stream_wait, or device_synchronize.
 * Does not affect stream_wait(false), and stream_wait(false) does not affect the skip state.
 */
EINSUMS_HOST EINSUMS_EXPORT void skip_next_wait();

/**
 * @brief Wait on the current thread's stream. Can be skippable or not.
 *
 * @param may_skip Indicate that the wait may be skipped to avoid unnecessary waits. Only skipped after a call to skip_next_wait,
 * then resets the skip flag so that later waits are not skipped.
 */
EINSUMS_HOST EINSUMS_EXPORT void stream_wait(bool may_skip = false);

/**
 * @brief Wait on all streams managed by Einsums.
 */
EINSUMS_HOST EINSUMS_EXPORT void all_stream_wait();

// Because I want it to be plural as well
#define all_streams_wait() all_stream_wait()

/**
 * @brief Gets the stream assigned to the current thread.
 */
EINSUMS_EXPORT EINSUMS_HOST hipStream_t get_stream();
EINSUMS_EXPORT EINSUMS_HOST hipStream_t get_stream(int thread_id);

/**
 * @brief Sets the stream assigned to the current thread.
 */
EINSUMS_EXPORT EINSUMS_HOST void set_stream(hipStream_t stream);
EINSUMS_EXPORT EINSUMS_HOST void set_stream(hipStream_t stream, int thread_id);

/**
 * @brief Synchronize to all operations on the device.
 *
 * Waits until the device is in an idle state before continuing. Blocks on all streams.
 */
EINSUMS_HOST inline void device_synchronize() {
    hip_catch(hipDeviceSynchronize());
}

/**
 * @brief Checks whether the parameter is exactly zero for its type.
 *
 * This is needed because of the lack of portability between types on the device.
 */
EINSUMS_DEVICE inline bool is_zero(double value) {
    return value == 0.0;
}

EINSUMS_DEVICE inline bool is_zero(float value) {
    return value == 0.0f;
}

EINSUMS_DEVICE inline bool is_zero(hipFloatComplex value) {
    return value.x == 0.0f && value.y == 0.0f;
}

EINSUMS_DEVICE inline bool is_zero(hipDoubleComplex value) {
    return value.x == 0.0 && value.y == 0.0;
}

/**
 * @brief Sets the input to zero for its type.
 *
 * This is needed because of the lack of portability between types on the device.
 */
EINSUMS_DEVICE inline void make_zero(double &value) {
    value = 0.0;
}

EINSUMS_DEVICE inline void make_zero(float &value) {
    value = 0.0f;
}

EINSUMS_DEVICE inline void make_zero(hipFloatComplex &value) {
    value.x = 0.0f;
    value.y = 0.0f;
}

EINSUMS_DEVICE inline void make_zero(hipDoubleComplex &value) {
    value.x = 0.0;
    value.y = 0.0;
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
EINSUMS_DEVICE inline void atomicAdd_wrap(float *address, float value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
EINSUMS_DEVICE inline void atomicAdd_wrap(double *address, double value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
EINSUMS_DEVICE inline void atomicAdd_wrap(hipFloatComplex *address, hipFloatComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
EINSUMS_DEVICE inline void atomicAdd_wrap(hipDoubleComplex *address, hipDoubleComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

} // namespace gpu
} // namespace einsums