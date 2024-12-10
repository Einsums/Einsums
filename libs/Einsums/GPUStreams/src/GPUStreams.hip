#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>

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

EINSUMS_HOST void initialize() {
    hip_catch(hipFree(nullptr));

    hip_catch(hipSetDevice(0));

    if (detail::default_streams.size() == 0) {
        detail::default_streams.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            hip_catch(hipStreamCreateWithFlags(&(detail::default_streams.at(i)), hipStreamNonBlocking));
        }
    }

    if (detail::skip_waits.size() == 0) {
        detail::skip_waits.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            detail::skip_waits.at(i) = false;
        }
    }

    if (detail::blas_handles.size() == 0) {
        detail::blas_handles.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            hipblas_catch(hipblasCreate(&(detail::blas_handles.at(i))));
            hipblas_catch(hipblasSetStream(detail::blas_handles.at(i), get_stream(i)));
        }
    }

    if (detail::solver_handles.size() == 0) {
        detail::solver_handles.resize(omp_get_max_threads());

        for (int i = 0; i < omp_get_max_threads(); i++) {
            hipsolver_catch(hipsolverCreate(&(detail::solver_handles.at(i))));
            hipsolver_catch(hipsolverSetStream(detail::solver_handles.at(i), get_stream(i)));
        }
    }
}

EINSUMS_HOST void finalize() {

    if (detail::default_streams.size() != 0) {
        for (int i = 0; i < detail::default_streams.size(); i++) {
            hip_catch(hipStreamDestroy(detail::default_streams.at(i)));
        }

        detail::default_streams.clear();
    }

    detail::skip_waits.clear();

    if (detail::blas_handles.size() != 0) {
        for (int i = 0; i < detail::blas_handles.size(); i++) {
            hipblas_catch(hipblasDestroy(detail::blas_handles.at(i)));
        }
        detail::blas_handles.clear();
    }

    if (detail::solver_handles.size() != 0) {
        for (int i = 0; i < detail::solver_handles.size(); i++) {
            hipsolver_catch(hipsolverDestroy(detail::solver_handles.at(i)));
        }
        detail::solver_handles.clear();
    }
}

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
    if (detail::default_streams.size() == 0) {
        initialize();
    }
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

    if (detail::default_streams.size() == 0) {
        initialize();
    }

    detail::default_streams.at(thread_id) = stream;

    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));
    hipsolver_catch(hipsolverSetStream(get_solver_handle(), stream));
}

EINSUMS_HOST int get_warpsize(void) {
    static int warpsize = 0;
    if (warpsize == 0) {
        hip_catch(hipDeviceGetAttribute(&warpsize, hipDeviceAttributeWarpSize, 0));
    }
    return warpsize;
}

EINSUMS_HOST hipblasHandle_t get_blas_handle(int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    if (detail::blas_handles.size() == 0) {
        initialize();
    }

    return detail::blas_handles.at(thread_id);
}

EINSUMS_HOST hipblasHandle_t get_blas_handle() {
    return get_blas_handle(omp_get_thread_num());
}

EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value, int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    if (detail::blas_handles.size() == 0) {
        initialize();
    }

    detail::blas_handles.at(thread_id) = value;
    return value;
}

EINSUMS_HOST hipblasHandle_t set_blas_handle(hipblasHandle_t value) {
    return set_blas_handle(value, omp_get_thread_num());
}

EINSUMS_HOST hipsolverHandle_t get_solver_handle(int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    if(detail::solver_handles.size() == 0) {
        initialize();
    }

    return detail::solver_handles.at(thread_id);
}

EINSUMS_HOST hipsolverHandle_t get_solver_handle() {
    return get_solver_handle(omp_get_thread_num());
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value, int thread_id) {
    auto guard = std::lock_guard(detail::wait_mutex);

    if(detail::solver_handles.size() == 0) {
        initialize();
    }

    detail::solver_handles.at(thread_id) = value;
    return detail::solver_handles.at(thread_id);
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value) {
    return set_solver_handle(value, omp_get_thread_num());
}

} // namespace gpu
} // namespace einsums