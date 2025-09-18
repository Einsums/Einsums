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

#if hipblasVersionMajor >= 3
using hipblasComplex = hipFloatComplex;
using hipblasDoubleComplex = hipDoubleComplex;
#endif

namespace einsums {
namespace gpu {

extern EINSUMS_EXPORT bool device_is_reset;

/**
 * @def get_worker_info
 * @brief Get the worker thread launch parameters on the GPU.
 *
 * @versionadded{1.0.0}
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
 *
 * @throws std::out_of_range If the current thread is too large. This should only happen if this function is called when the list of streams
 * is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t get_blas_handle();

/**
 * @brief Get the internal hipBLAS handle for the requested thread.
 *
 * @param[in] thread_id The id of the thread being requested.
 *
 * @return The current internal hipBLAS handle.
 *
 * @throws std::out_of_range If the parameter is larger than the number of threads or is negative. It may also be thrown if the list of
 * streams is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t get_blas_handle(int thread_id);

/**
 * @brief Get the internal hipSolver handle.
 *
 * @return The current internal hipSolver handle.
 *
 * @throws std::out_of_range If the current thread is too large. This should only happen if this function is called when the list of streams
 * is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t get_solver_handle();

/**
 * @brief Get the internal hipSolver handle for the requested thread..
 *
 * @param[in] thread_id The id of the thread being requested.
 *
 * @return The current internal hipSolver handle.
 *
 * @throws std::out_of_range If the parameter is larger than the number of threads or is negative. It may also be thrown if the list of
 * streams is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t get_solver_handle(int thread_id);

// Set the handles used internally. Useful to avoid creating multiple contexts.
/**
 * @brief Set the internal hipBLAS handle.
 *
 * @param[in] value The new handle.
 *
 * @throws std::out_of_range If the current thread is too large. This should only happen if this function is called when the list of streams
 * is uninitialized.
 *
 * @return The new handle.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value);

/**
 * @brief Set the internal hipBLAS handle for the requested thread.
 *
 * @param[in] value The new handle.
 * @param[in] thread_id The id of the thread to modify.
 *
 * @return The new handle.
 *
 * @throws std::out_of_range If the thread parameter is larger than the number of threads or is negative. It may also be thrown if the list
 * of streams is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value, int thread_id);

/**
 * @brief Set the internal hipSolver handle.
 *
 * @param[in] value The new handle.
 *
 * @return The new handle.
 *
 * @throws std::out_of_range If the current thread is too large. This should only happen if this function is called when the list of streams
 * is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t set_solver_handle(hipsolverHandle_t value);

/**
 * @brief Set the internal hipSolver handle for the requested thread.
 *
 * @param[in] value The new handle.
 * @param[in] thread_id The id of the thread to modify.
 *
 * @return The new handle.
 *
 * @throws std::out_of_range If the thread parameter is larger than the number of threads or is negative. It may also be thrown if the list
 * of streams is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipsolverHandle_t set_solver_handle(hipsolverHandle_t value, int thread_id);

/**
 * Get an appropriate block size for a kernel.
 *
 * @param[in] compute_size The number of elements being processed.
 *
 * @versionadded{1.0.0}
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
 *
 * @param[in] compute_size The number of elements being processed.
 *
 * @versionadded{1.0.0}
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
 *
 * @param[in] stream The stream to wait on.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void stream_wait(hipStream_t stream);

/**
 * @brief Indicates that the next skippable wait on the current thread should be skipped.
 * Does not apply to stream_wait(stream) with the stream specified, all_stream_wait, or device_synchronize.
 * Does not affect stream_wait(false), and stream_wait(false) does not affect the skip state.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void skip_next_wait();

/**
 * @brief Wait on the current thread's stream. Can be skippable or not.
 *
 * @param[in] may_skip Indicate that the wait may be skipped to avoid unnecessary waits. Only skipped after a call to skip_next_wait,
 * then resets the skip flag so that later waits are not skipped.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void stream_wait(bool may_skip = false);

/**
 * @brief Wait on all streams managed by Einsums.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void all_stream_wait();

/**
 * @brief Gets the stream assigned to the current thread.
 *
 * @return The stream associated with the current thread.
 *
 * @throws std::out_of_range If the current thread is too large. This should only happen if this function is called when the list of streams
 * is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipStream_t get_stream();

/**
 * @brief Gets the stream assigned to the requested thread.
 *
 * @param[in] thread_id The thread to query.
 *
 * @return The stream associated with the thread.
 *
 * @throws std::out_of_range If the parameter is larger than the number of threads or is negative. It may also be thrown if the list of
 * streams is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST hipStream_t get_stream(int thread_id);

/**
 * @brief Sets the stream assigned to the current thread.
 *
 * @param[in] stream The new stream.
 *
 * @throws std::out_of_range If the current thread is too large. This should only happen if this function is called when the list of streams
 * is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST void set_stream(hipStream_t stream);

/**
 * @brief Sets the stream assigned to the requested thread.
 *
 * @param[in] stream The new stream.
 * @param[in] thread_id The ID of the thread to modify.
 *
 * @throws std::out_of_range If the thread parameter is larger than the number of threads or is negative. It may also be thrown if the list
 * of streams is uninitialized.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST void set_stream(hipStream_t stream, int thread_id);

/**
 * @brief Synchronize to all operations on the device.
 *
 * Waits until the device is in an idle state before continuing. Blocks on all streams.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST inline void device_synchronize() {
    hip_catch(hipDeviceSynchronize());
}

/**
 * @brief Checks whether the parameter is exactly zero for its type.
 *
 * This is needed because of the lack of portability between types on the device.
 *
 * @param[in] value The value to check.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_DEVICE constexpr bool is_zero(double value) noexcept {
    return value == 0.0;
}

/// @copydoc is_zero(double)
EINSUMS_DEVICE constexpr bool is_zero(float value) noexcept {
    return value == 0.0f;
}

/// @copydoc is_zero(double)
EINSUMS_DEVICE constexpr bool is_zero(hipFloatComplex value) noexcept {
    return value.x == 0.0f && value.y == 0.0f;
}

/// @copydoc is_zero(double)
EINSUMS_DEVICE constexpr bool is_zero(hipDoubleComplex value) noexcept {
    return value.x == 0.0 && value.y == 0.0;
}

/**
 * @brief Sets the input to zero for its type.
 *
 * This is needed because of the lack of portability between types on the device.
 *
 * @param[out] value A reference to the value to change.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_DEVICE inline void make_zero(double &value) noexcept {
    value = 0.0;
}

/// @copydoc make_zero(double&)
EINSUMS_DEVICE inline void make_zero(float &value) noexcept {
    value = 0.0f;
}

/// @copydoc make_zero(double&)
EINSUMS_DEVICE inline void make_zero(hipFloatComplex &value) noexcept {
    value.x = 0.0f;
    value.y = 0.0f;
}

/// @copydoc make_zero(double&)
EINSUMS_DEVICE inline void make_zero(hipDoubleComplex &value) noexcept {
    value.x = 0.0;
    value.y = 0.0;
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 *
 * @param[out] address The destination address.
 * @param[in] value The value to add.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_DEVICE inline void atomicAdd_wrap(float *address, float value) noexcept {
    atomicAdd(address, value);
}

/// @copydoc atomicAdd_wrap(float*,float)
EINSUMS_DEVICE inline void atomicAdd_wrap(double *address, double value) noexcept {
    atomicAdd(address, value);
}

/// @copydoc atomicAdd_wrap(float*,float)
EINSUMS_DEVICE inline void atomicAdd_wrap(hipFloatComplex *address, hipFloatComplex value) noexcept {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

/// @copydoc atomicAdd_wrap(float*,float)
EINSUMS_DEVICE inline void atomicAdd_wrap(hipDoubleComplex *address, hipDoubleComplex value) noexcept {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

} // namespace gpu
} // namespace einsums