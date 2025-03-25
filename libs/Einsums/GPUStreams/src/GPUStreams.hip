//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/GPUStreams/InitModule.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <mutex>
#include <omp.h>
#include <vector>

namespace einsums {
namespace gpu {

namespace detail {
std::vector<hipblasHandle_t>   blas_handles;
std::vector<hipsolverHandle_t> solver_handles;
std::vector<hipStream_t>       default_streams;
std::vector<bool>              skip_waits;
std::recursive_mutex           wait_mutex;
} // namespace detail
} // namespace gpu
} // namespace einsums

EINSUMS_HOST int einsums::setup_Einsums_GPUStreams() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_startup_function(einsums::initialize_Einsums_GPUStreams);
        einsums::register_shutdown_function(einsums::finalize_Einsums_GPUStreams);

        is_initialized = true;
    }

    return 0;
}

EINSUMS_HOST void einsums::initialize_Einsums_GPUStreams() {
    using namespace einsums::gpu;
    hip_catch(hipFree(nullptr));

    hip_catch(hipSetDevice(0));

    if (gpu::detail::default_streams.size() == 0) {
        gpu::detail::default_streams.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            hip_catch(hipStreamCreateWithFlags(&(gpu::detail::default_streams.at(i)), hipStreamNonBlocking));
        }
    }

    if (gpu::detail::skip_waits.size() == 0) {
        gpu::detail::skip_waits.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            gpu::detail::skip_waits.at(i) = false;
        }
    }

    if (gpu::detail::blas_handles.size() == 0) {
        gpu::detail::blas_handles.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            hipblas_catch(hipblasCreate(&(gpu::detail::blas_handles.at(i))));
            hipblas_catch(hipblasSetStream(gpu::detail::blas_handles.at(i), get_stream(i)));
        }
    }

    if (gpu::detail::solver_handles.size() == 0) {
        gpu::detail::solver_handles.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            hipsolver_catch(hipsolverCreate(&(gpu::detail::solver_handles.at(i))));
            hipsolver_catch(hipsolverSetStream(gpu::detail::solver_handles.at(i), get_stream(i)));
        }
    }
}

EINSUMS_HOST void einsums::finalize_Einsums_GPUStreams() {
    using namespace einsums::gpu;
    if (gpu::detail::default_streams.size() != 0) {
        for (int i = 0; i < gpu::detail::default_streams.size(); i++) {
            hip_catch(hipStreamDestroy(gpu::detail::default_streams.at(i)));
        }

        gpu::detail::default_streams.clear();
    }

    gpu::detail::skip_waits.clear();

    if (gpu::detail::blas_handles.size() != 0) {
        for (int i = 0; i < gpu::detail::blas_handles.size(); i++) {
            hipblas_catch(hipblasDestroy(gpu::detail::blas_handles.at(i)));
        }
        gpu::detail::blas_handles.clear();
    }

    if (gpu::detail::solver_handles.size() != 0) {
        for (int i = 0; i < gpu::detail::solver_handles.size(); i++) {
            hipsolver_catch(hipsolverDestroy(gpu::detail::solver_handles.at(i)));
        }
        gpu::detail::solver_handles.clear();
    }

    hip_catch(hipDeviceReset());
}

namespace einsums {
namespace gpu {

EINSUMS_HOST void skip_next_wait() {
    detail::skip_waits.at(omp_get_thread_num()) = true;
}

EINSUMS_HOST void stream_wait(bool may_skip) {
    if (may_skip && detail::skip_waits.at(omp_get_thread_num())) {
        detail::skip_waits.at(omp_get_thread_num()) = false;
        return;
    }
    stream_wait(get_stream());
}

EINSUMS_HOST void stream_wait(hipStream_t stream) {
    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));

    detail::wait_mutex.lock();

    hip_catch(hipEventRecord(wait_event, stream));

    detail::wait_mutex.unlock();

    hip_catch(hipEventSynchronize(wait_event));
    hip_catch(hipEventDestroy(wait_event));
}

EINSUMS_HOST void all_stream_wait() {

    std::vector<hipEvent_t> wait_events(omp_get_max_threads());

    detail::wait_mutex.lock();

    for (int i = 0; i < wait_events.size(); i++) {
        hip_catch(hipEventCreate(&(wait_events[i])));
        hip_catch(hipEventRecord(wait_events[i], detail::default_streams[i]));
    }

    detail::wait_mutex.unlock();

    for (int i = 0; i < wait_events.size(); i++) {
        hip_catch(hipEventSynchronize(wait_events[i]));
        hip_catch(hipEventDestroy(wait_events[i]));
    }
}

/**
 * @brief Gets the stream assigned to the current thread.
 */
EINSUMS_HOST hipStream_t get_stream() {
    return get_stream(omp_get_thread_num());
}

EINSUMS_HOST hipStream_t get_stream(int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);
    return detail::default_streams.at(thread_id);
}

/**
 * @brief Sets the stream assigned to the current thread.
 */
EINSUMS_HOST void set_stream(hipStream_t stream) {
    set_stream(stream, omp_get_thread_num());
}

EINSUMS_HOST void set_stream(hipStream_t stream, int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    detail::default_streams.at(thread_id) = stream;

    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));
    hipsolver_catch(hipsolverSetStream(get_solver_handle(), stream));
}

// EINSUMS_HOST int get_warpsize(void) {
//     static int warpsize = 0;
//     if (warpsize == 0) {
//         hip_catch(hipDeviceGetAttribute(&warpsize, hipDeviceAttributeWarpSize, 0));
//     }
//     return warpsize;
// }

EINSUMS_HOST hipblasHandle_t get_blas_handle(int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    return detail::blas_handles.at(thread_id);
}

EINSUMS_HOST hipblasHandle_t get_blas_handle() {
    return get_blas_handle(omp_get_thread_num());
}

EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value, int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    detail::blas_handles.at(thread_id) = value;
    return value;
}

EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value) {
    return set_blas_handle(value, omp_get_thread_num());
}

EINSUMS_HOST hipsolverHandle_t get_solver_handle(int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    return detail::solver_handles.at(thread_id);
}

EINSUMS_HOST hipsolverHandle_t get_solver_handle() {
    return get_solver_handle(omp_get_thread_num());
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value, int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    detail::solver_handles.at(thread_id) = value;
    return detail::solver_handles.at(thread_id);
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value) {
    return set_solver_handle(value, omp_get_thread_num());
}

} // namespace gpu
} // namespace einsums