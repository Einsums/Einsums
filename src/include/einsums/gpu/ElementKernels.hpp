#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums {
namespace element_operations {
namespace detail {

template <typename T, size_t Rank>
__global__ void sum_kernel(T *out, const T *data, size_t *dims, size_t *strides, size_t size) {
    extern __shared__ T work_array[];

    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    work_array[worker] = T{0};

    if (worker == 0) {
        *out = T{0};
    }

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }

        work_array[worker] += data[index];
    }
    __syncthreads();

    int log = 30 - __clz(kernel_size);

    for (int s = 0; s < log; s++) {
        size_t index = worker << (s + 1);
        size_t next  = index + (1 << s);
        if (index < kernel_size && next < kernel_size) {
            work_array[index] += work_array[next];
        }
        __syncthreads();
    }

    if (worker == 0) {
        *out = work_array[0];
    }
}

template <typename T, size_t Rank>
__global__ void max_kernel(T *out, const T *data, size_t *dims, size_t *strides, size_t size) {
    extern __shared__ T work_array[];

    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    work_array[worker] = T{-INFINITY};

    if (worker == 0) {
        *out = T{0};
    }

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }

        if (data[index] > work_array[worker]) {
            work_array[worker] = data[index];
        }
    }

    __syncthreads();

    int log = 30 - __clz(kernel_size);

    for (int s = 0; s < log; s++) {
        size_t index = worker << (s + 1);
        size_t next  = index + (1 << s);
        if (index < kernel_size && next < kernel_size) {
            work_array[index] = (work_array[index] > work_array[next]) ? work_array[index] : work_array[next];
        }
        __syncthreads();
    }

    if (worker == 0) {
        *out = work_array[0];
    }
}

template <typename T, size_t Rank>
__global__ void min_kernel(T *out, const T *data, size_t *dims, size_t *strides, size_t size) {
    extern __shared__ T work_array[];

    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    work_array[worker] = T{INFINITY};

    if (worker == 0) {
        *out = T{0};
    }

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }

        if (data[index] < work_array[worker]) {
            work_array[worker] = data[index];
        }
    }

    __syncthreads();

    int log = 30 - __clz(kernel_size);

    for (int s = 0; s < log; s++) {
        size_t index = worker << (s + 1);
        size_t next  = index + (1 << s);
        if (index < kernel_size && next < kernel_size) {
            work_array[index] = (work_array[index] < work_array[next]) ? work_array[index] : work_array[next];
        }
        __syncthreads();
    }

    if (worker == 0) {
        *out = work_array[0];
    }
}

template <typename T, size_t Rank>
__global__ void abs_kernel(T *data, size_t *dims, size_t *strides, size_t size) {
    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }

        if constexpr (std::is_same_v<T, float>) {
            data[index] = fabsf(data[index]);
        } else if constexpr (std::is_same_v<T, double>) {
            data[index] = fabs(data[index]);
        } else if constexpr (std::is_same_v<T, hipComplex>) {
            data[index] = hypotf(data[index].x, data[index].y);
        } else if constexpr (std::is_same_v<T, hipDoubleComplex>) {
            data[index] = hypot(data[index].x, data[index].y);
        } else {
            data[index] = (data[index] < 0) ? -data[index] : data[index]; // Fallback.
        }
    }
}

template <typename T, size_t Rank>
__global__ void invert_kernel(T *data, size_t *dims, size_t *strides, size_t size) {
    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }

        data[index] = T{1} / data[index];
    }
}

template <typename T, size_t Rank>
__global__ void exp_kernel(T *data, size_t *dims, size_t *strides, size_t size) {
    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }
#ifdef EINSUMS_FAST_INTRINSICS
        if constexpr (std::is_same_v<T, float>) {
            data[index] = __expf(data[index]);
        } else if constexpr (std::is_same_v<T, double>) {
            data[index] = exp(data[index]);
        } else if constexpr (std::is_same_v<T, hipComplex>) {
            float scale   = __expf(data[index].x);
            data[index].x = scale * __cosf(data[index].y);
            data[index].y = scale * __sinf(data[index].y);
        } else if constexpr (std::is_same_v<T, hipDoubleComplex>) {
            double scale  = exp(data[index].x);
            data[index].x = scale * cos(data[index].y);
            data[index].y = scale * sin(data[index].y);
        }
#else
        if constexpr (std::is_same_v<T, float>) {
            data[index] = expf(data[index]);
        } else if constexpr (std::is_same_v<T, double>) {
            data[index] = exp(data[index]);
        } else if constexpr (std::is_same_v<T, hipComplex>) {
            float scale   = expf(data[index].x);
            data[index].x = scale * cosf(data[index].y);
            data[index].y = scale * sinf(data[index].y);
        } else if constexpr (std::is_same_v<T, hipDoubleComplex>) {
            double scale  = exp(data[index].x);
            data[index].x = scale * cos(data[index].y);
            data[index].y = scale * sin(data[index].y);
        }
#endif
    }
}

template <typename T, size_t Rank>
__global__ void scale_kernel(T *data, T scale, size_t *dims, size_t *strides, size_t size) {
    unsigned int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    for (ssize_t sentinel = worker; sentinel < size; sentinel += kernel_size) {
        size_t index = 0, quotient = sentinel, remainder = 0;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            remainder = quotient % dims[i];
            quotient /= dims[i];
            index += strides[i] * remainder;
        }

        data[index] = scale * data[index];
    }
}

} // namespace detail
} // namespace element_operations
} // namespace einsums