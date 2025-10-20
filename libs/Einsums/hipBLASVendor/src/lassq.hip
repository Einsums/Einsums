//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <cstddef>
#include <exception>
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {

__global__ void slassq_kernel(int64_t n, float const *__restrict__ x, int64_t incx, float *__restrict__ big_sum_all,
                              float *__restrict__ medium_sum_all, float *__restrict__ small_sum_all) {
    constexpr float big_thresh   = 4503599600000000.0f;
    constexpr float small_thresh = 1.0842022e-19f;
    constexpr float big_scale    = 2.646978e-23f;
    constexpr float small_scale  = 3.7778932e+22;
    // extern __shared__ float shared[];

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;
    int warp_id   = thread_id / warpSize;

    int num_warps = num_threads / warpSize;

    int64_t parallel_elems  = n / num_threads;
    int64_t remaining_elems = n % num_threads;

    float small_sum  = 0.0f;
    float medium_sum = 0.0f;
    float big_sum    = 0.0f;

    // float const *x_curr = x + (ptrdiff_t)warp_id * warpSize * parallel_elems * incx;

    // Do the things.
    for (int64_t i = inwarp_id; i < parallel_elems * warpSize; i += incx * warpSize) {
        // shared[thread_id] += x[i + warp_id * warpSize * parallel_elems * incx];
        float ax = fabsf(x[i + warp_id * warpSize * parallel_elems * incx]);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    if (thread_id < remaining_elems) {
        // shared[thread_id] += x[thread_id + num_warps * warpSize * parallel_elems * incx];
        float ax = fabsf(x[thread_id + num_warps * warpSize * parallel_elems * incx]);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
    // #pragma unroll
    //     for (uint32_t i = warpSize / 2; i >= 1; i /= 2) {
    //         medium_sum = medium_sum + __shfl_down(medium_sum, i + warpSize);
    //     }

    // Combine the sums.
    // if (inwarp_id == 0) {
    atomicAdd(big_sum_all, big_sum);
    atomicAdd(medium_sum_all, medium_sum);
    atomicAdd(small_sum_all, small_sum);
    // }
}

__global__ void dlassq_kernel(int64_t n, double const *__restrict__ x, int64_t incx, double *__restrict__ big_sum_all,
                              double *__restrict__ medium_sum_all, double *__restrict__ small_sum_all) {
    constexpr double big_thresh   = 9.989595361011175e+145;
    constexpr double small_thresh = 1.4916681462400413e-154;
    constexpr double big_scale    = 1.1113793747425387e-162;
    constexpr double small_scale  = 8.997827589086393e+161;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;
    int warp_id   = thread_id / warpSize;

    int num_warps = num_threads / warpSize;

    int64_t parallel_elems  = n / num_threads;
    int64_t remaining_elems = n % num_threads;

    double small_sum  = 0.0f;
    double medium_sum = 0.0f;
    double big_sum    = 0.0f;

    // Do the things.
    for (int64_t i = inwarp_id; i < parallel_elems * warpSize; i += incx * warpSize) {
        double ax = fabs(x[i + warp_id * warpSize * parallel_elems * incx]);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    if (thread_id < remaining_elems) {
        double ax = fabs(x[thread_id + num_warps * warpSize * parallel_elems * incx]);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
    // #pragma unroll
    //     for (uint32_t i = warpSize / 2; i >= 1; i /= 2) {
    //         medium_sum = medium_sum + __shfl_down(medium_sum, i + warpSize);
    //     }

    // Combine the sums.
    // if (inwarp_id == 0) {
    atomicAdd(big_sum_all, big_sum);
    atomicAdd(medium_sum_all, medium_sum);
    atomicAdd(small_sum_all, small_sum);
    // }
}

__global__ void classq_kernel(int64_t n, hipComplex const *__restrict__ x, int64_t incx, float *__restrict__ big_sum_all,
                              float *__restrict__ medium_sum_all, float *__restrict__ small_sum_all) {

    constexpr float big_thresh   = 4503599600000000.0f;
    constexpr float small_thresh = 1.0842022e-19f;
    constexpr float big_scale    = 2.646978e-23f;
    constexpr float small_scale  = 3.7778932e+22;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;
    int warp_id   = thread_id / warpSize;

    int num_warps = num_threads / warpSize;

    int64_t parallel_elems  = n / num_threads;
    int64_t remaining_elems = n % num_threads;

    float small_sum  = 0.0f;
    float medium_sum = 0.0f;
    float big_sum    = 0.0f;

    // float const *x_curr = x + (ptrdiff_t)warp_id * warpSize * parallel_elems * incx;

    // Do the things.
    for (int64_t i = inwarp_id; i < parallel_elems * warpSize; i += incx * warpSize) {
        // shared[thread_id] += x[i + warp_id * warpSize * parallel_elems * incx];
        float ax = fabsf(x[i + warp_id * warpSize * parallel_elems * incx].x);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }

        ax = fabsf(x[i + warp_id * warpSize * parallel_elems * incx].y);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    if (thread_id < remaining_elems) {
        // shared[thread_id] += x[thread_id + num_warps * warpSize * parallel_elems * incx];
        float ax = fabsf(x[thread_id + num_warps * warpSize * parallel_elems * incx].x);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }

        ax = fabsf(x[thread_id + num_warps * warpSize * parallel_elems * incx].y);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
    // #pragma unroll
    //     for (uint32_t i = warpSize / 2; i >= 1; i /= 2) {
    //         medium_sum = medium_sum + __shfl_down(medium_sum, i + warpSize);
    //     }

    // Combine the sums.
    // if (inwarp_id == 0) {
    atomicAdd(big_sum_all, big_sum);
    atomicAdd(medium_sum_all, medium_sum);
    atomicAdd(small_sum_all, small_sum);
    // }
}

__global__ void zlassq_kernel(int64_t n, hipDoubleComplex const *__restrict__ x, int64_t incx, double *__restrict__ big_sum_all,
                              double *__restrict__ medium_sum_all, double *__restrict__ small_sum_all) {
    constexpr double big_thresh   = 9.989595361011175e+145;
    constexpr double small_thresh = 1.4916681462400413e-154;
    constexpr double big_scale    = 1.1113793747425387e-162;
    constexpr double small_scale  = 8.997827589086393e+161;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;
    int warp_id   = thread_id / warpSize;

    int num_warps = num_threads / warpSize;

    int64_t parallel_elems  = n / num_threads;
    int64_t remaining_elems = n % num_threads;

    double small_sum  = 0.0;
    double medium_sum = 0.0;
    double big_sum    = 0.0;

    // float const *x_curr = x + (ptrdiff_t)warp_id * warpSize * parallel_elems * incx;

    // Do the things.
    for (int64_t i = inwarp_id; i < parallel_elems * warpSize; i += incx * warpSize) {
        // shared[thread_id] += x[i + warp_id * warpSize * parallel_elems * incx];
        double ax = fabs(x[i + warp_id * warpSize * parallel_elems * incx].x);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }

        ax = fabs(x[i + warp_id * warpSize * parallel_elems * incx].y);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    if (thread_id < remaining_elems) {
        // shared[thread_id] += x[thread_id + num_warps * warpSize * parallel_elems * incx];
        double ax = fabs(x[thread_id + num_warps * warpSize * parallel_elems * incx].x);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }

        ax = fabs(x[thread_id + num_warps * warpSize * parallel_elems * incx].y);
        if (ax > big_thresh) {
            big_sum += (ax * big_scale) * (ax * big_scale);
        } else if (ax < small_thresh) {
            small_sum += (ax * small_scale) * (ax * small_scale);
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
    // #pragma unroll
    //     for (uint32_t i = warpSize / 2; i >= 1; i /= 2) {
    //         medium_sum = medium_sum + __shfl_down(medium_sum, i + warpSize);
    //     }

    // Combine the sums.
    // if (inwarp_id == 0) {
    atomicAdd(big_sum_all, big_sum);
    atomicAdd(medium_sum_all, medium_sum);
    atomicAdd(small_sum_all, small_sum);
    // }
}

void slassq(int64_t n, float const *x, int64_t incx, float *scale, float *sumsq) {
    LabeledSection("slassq");

    if (std::isnan(*scale) || std::isnan(*sumsq)) {
        return;
    }

    constexpr float big_thresh   = 4503599600000000.0f;
    constexpr float small_thresh = 1.0842022e-19f;
    constexpr float big_scale    = 2.646978e-23f;
    constexpr float small_scale  = 3.7778932e+22;

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<float>();

    auto work = gpu_alloc.allocate(3);

    hip_catch(hipMemset(work.get(), 0, 3 * sizeof(float)));

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    slassq_kernel<<<block_size, blocks, 0, stream>>>(n, x, incx, work, work + 1, work + 2);

    gpu::stream_wait();

    float big_sum_all, medium_sum_all, small_sum_all;

    std::memcpy(&big_sum_all, work, sizeof(float));
    std::memcpy(&medium_sum_all, work + 1, sizeof(float));
    std::memcpy(&small_sum_all, work + 2, sizeof(float));

    // Consider the scale and sum from the input.
    if (*sumsq > 0.0f) {
        float ax = *scale * std::sqrt(*sumsq);
        if (ax > big_thresh) {
            if (*scale > 1.0f) {
                *scale *= big_scale;
                big_sum_all += *scale * *scale * *sumsq;
            } else {
                big_sum_all += *scale * (*scale * (big_scale * (big_scale * *sumsq)));
            }
        } else if (ax < small_thresh) {
            if (*scale < 1.0f) {
                *scale *= small_scale;
                small_sum_all += *scale * *scale * *sumsq;
            } else {
                small_sum_all += *scale * (*scale * (small_scale * (small_scale * *sumsq)));
            }
        } else {
            medium_sum_all += *scale * (*scale * *sumsq);
        }
    }

    if (big_sum_all > 0.0f) {
        if (medium_sum_all > 0.0f || std::isnan(medium_sum_all)) {
            big_sum_all += (medium_sum_all * big_scale) * big_scale;
        }
        *scale = 1.0f / big_scale;
        *sumsq = big_sum_all;
    } else if (small_sum_all > 0.0f) {
        float ymin, ymax;
        if (medium_sum_all > 0.0f || std::isnan(medium_sum_all)) {
            medium_sum_all = std::sqrt(medium_sum_all);
            small_sum_all  = std::sqrt(small_sum_all) / small_scale;
            if (small_sum_all > medium_sum_all) {
                ymin = medium_sum_all;
                ymax = small_sum_all;
            } else {
                ymin = small_sum_all;
                ymax = medium_sum_all;
            }

            *scale = 1.0f;
            *sumsq = ymax * ymax * (1.0f + (ymin / ymax) * (ymin / ymax));
        } else {
            *scale = 1.0f / small_scale;
            *sumsq = small_sum_all;
        }
    } else {
        *scale = 1.0f;
        *sumsq = medium_sum_all;
    }

    gpu_alloc.deallocate(work, 3);
}

void dlassq(int64_t n, double const *x, int64_t incx, double *scale, double *sumsq) {
    LabeledSection("dlassq");

    if (std::isnan(*scale) || std::isnan(*sumsq)) {
        return;
    }

    constexpr double big_thresh   = 9.989595361011175e+145;
    constexpr double small_thresh = 1.4916681462400413e-154;
    constexpr double big_scale    = 1.1113793747425387e-162;
    constexpr double small_scale  = 8.997827589086393e+161;

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<double>();

    auto work = gpu_alloc.allocate(3);

    hip_catch(hipMemset(work.get(), 0, 3 * sizeof(double)));

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    dlassq_kernel<<<block_size, blocks, 0, stream>>>(n, x, incx, work, work + 1, work + 2);

    gpu::stream_wait();

    double big_sum_all, medium_sum_all, small_sum_all;

    std::memcpy(&big_sum_all, work, sizeof(double));
    std::memcpy(&medium_sum_all, work + 1, sizeof(double));
    std::memcpy(&small_sum_all, work + 2, sizeof(double));

    // Consider the scale and sum from the input.
    if (*sumsq > 0.0) {
        double ax = *scale * std::sqrt(*sumsq);
        if (ax > big_thresh) {
            if (*scale > 1.0) {
                *scale *= big_scale;
                big_sum_all += *scale * *scale * *sumsq;
            } else {
                big_sum_all += *scale * (*scale * (big_scale * (big_scale * *sumsq)));
            }
        } else if (ax < small_thresh) {
            if (*scale < 1.0f) {
                *scale *= small_scale;
                small_sum_all += *scale * *scale * *sumsq;
            } else {
                small_sum_all += *scale * (*scale * (small_scale * (small_scale * *sumsq)));
            }
        } else {
            medium_sum_all += *scale * (*scale * *sumsq);
        }
    }

    if (big_sum_all > 0.0) {
        if (medium_sum_all > 0.0 || std::isnan(medium_sum_all)) {
            big_sum_all += (medium_sum_all * big_scale) * big_scale;
        }
        *scale = 1.0 / big_scale;
        *sumsq = big_sum_all;
    } else if (small_sum_all > 0.0) {
        double ymin, ymax;
        if (medium_sum_all > 0.0 || std::isnan(medium_sum_all)) {
            medium_sum_all = std::sqrt(medium_sum_all);
            small_sum_all  = std::sqrt(small_sum_all) / small_scale;
            if (small_sum_all > medium_sum_all) {
                ymin = medium_sum_all;
                ymax = small_sum_all;
            } else {
                ymin = small_sum_all;
                ymax = medium_sum_all;
            }

            *scale = 1.0;
            *sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            *scale = 1.0 / small_scale;
            *sumsq = small_sum_all;
        }
    } else {
        *scale = 1.0;
        *sumsq = medium_sum_all;
    }

    gpu_alloc.deallocate(work, 3);
}

void classq(int64_t n, std::complex<float> const *x, int64_t incx, float *scale, float *sumsq) {
    LabeledSection("classq");

    if (std::isnan(*scale) || std::isnan(*sumsq)) {
        return;
    }

    constexpr float big_thresh   = 4503599600000000.0f;
    constexpr float small_thresh = 1.0842022e-19f;
    constexpr float big_scale    = 2.646978e-23f;
    constexpr float small_scale  = 3.7778932e+22;

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<float>();

    auto work = gpu_alloc.allocate(3);

    hip_catch(hipMemset(work.get(), 0, 3 * sizeof(float)));

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    classq_kernel<<<block_size, blocks, 0, stream>>>(n, (hipComplex *)x, incx, work, work + 1, work + 2);

    gpu::stream_wait();

    float big_sum_all, medium_sum_all, small_sum_all;

    std::memcpy(&big_sum_all, work, sizeof(float));
    std::memcpy(&medium_sum_all, work + 1, sizeof(float));
    std::memcpy(&small_sum_all, work + 2, sizeof(float));

    // Consider the scale and sum from the input.
    if (*sumsq > 0.0f) {
        float ax = *scale * std::sqrt(*sumsq);
        if (ax > big_thresh) {
            if (*scale > 1.0f) {
                *scale *= big_scale;
                big_sum_all += *scale * *scale * *sumsq;
            } else {
                big_sum_all += *scale * (*scale * (big_scale * (big_scale * *sumsq)));
            }
        } else if (ax < small_thresh) {
            if (*scale < 1.0f) {
                *scale *= small_scale;
                small_sum_all += *scale * *scale * *sumsq;
            } else {
                small_sum_all += *scale * (*scale * (small_scale * (small_scale * *sumsq)));
            }
        } else {
            medium_sum_all += *scale * (*scale * *sumsq);
        }
    }

    if (big_sum_all > 0.0f) {
        if (medium_sum_all > 0.0f || std::isnan(medium_sum_all)) {
            big_sum_all += (medium_sum_all * big_scale) * big_scale;
        }
        *scale = 1.0f / big_scale;
        *sumsq = big_sum_all;
    } else if (small_sum_all > 0.0f) {
        float ymin, ymax;
        if (medium_sum_all > 0.0f || std::isnan(medium_sum_all)) {
            medium_sum_all = std::sqrt(medium_sum_all);
            small_sum_all  = std::sqrt(small_sum_all) / small_scale;
            if (small_sum_all > medium_sum_all) {
                ymin = medium_sum_all;
                ymax = small_sum_all;
            } else {
                ymin = small_sum_all;
                ymax = medium_sum_all;
            }

            *scale = 1.0f;
            *sumsq = ymax * ymax * (1.0f + (ymin / ymax) * (ymin / ymax));
        } else {
            *scale = 1.0f / small_scale;
            *sumsq = small_sum_all;
        }
    } else {
        *scale = 1.0f;
        *sumsq = medium_sum_all;
    }

    gpu_alloc.deallocate(work, 3);
}

void zlassq(int64_t n, std::complex<double> const *x, int64_t incx, double *scale, double *sumsq) {
    LabeledSection("zlassq");

    if (std::isnan(*scale) || std::isnan(*sumsq)) {
        return;
    }

    constexpr double big_thresh   = 9.989595361011175e+145;
    constexpr double small_thresh = 1.4916681462400413e-154;
    constexpr double big_scale    = 1.1113793747425387e-162;
    constexpr double small_scale  = 8.997827589086393e+161;

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<double>();

    auto work = gpu_alloc.allocate(3);

    hip_catch(hipMemset(work.get(), 0, 3 * sizeof(double)));

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    zlassq_kernel<<<block_size, blocks, 0, stream>>>(n, (hipDoubleComplex *)x, incx, work, work + 1, work + 2);

    gpu::stream_wait();

    double big_sum_all, medium_sum_all, small_sum_all;

    std::memcpy(&big_sum_all, work, sizeof(double));
    std::memcpy(&medium_sum_all, work + 1, sizeof(double));
    std::memcpy(&small_sum_all, work + 2, sizeof(double));

    // Consider the scale and sum from the input.
    if (*sumsq > 0.0) {
        double ax = *scale * std::sqrt(*sumsq);
        if (ax > big_thresh) {
            if (*scale > 1.0) {
                *scale *= big_scale;
                big_sum_all += *scale * *scale * *sumsq;
            } else {
                big_sum_all += *scale * (*scale * (big_scale * (big_scale * *sumsq)));
            }
        } else if (ax < small_thresh) {
            if (*scale < 1.0f) {
                *scale *= small_scale;
                small_sum_all += *scale * *scale * *sumsq;
            } else {
                small_sum_all += *scale * (*scale * (small_scale * (small_scale * *sumsq)));
            }
        } else {
            medium_sum_all += *scale * (*scale * *sumsq);
        }
    }

    if (big_sum_all > 0.0) {
        if (medium_sum_all > 0.0 || std::isnan(medium_sum_all)) {
            big_sum_all += (medium_sum_all * big_scale) * big_scale;
        }
        *scale = 1.0 / big_scale;
        *sumsq = big_sum_all;
    } else if (small_sum_all > 0.0) {
        double ymin, ymax;
        if (medium_sum_all > 0.0 || std::isnan(medium_sum_all)) {
            medium_sum_all = std::sqrt(medium_sum_all);
            small_sum_all  = std::sqrt(small_sum_all) / small_scale;
            if (small_sum_all > medium_sum_all) {
                ymin = medium_sum_all;
                ymax = small_sum_all;
            } else {
                ymin = small_sum_all;
                ymax = medium_sum_all;
            }

            *scale = 1.0;
            *sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            *scale = 1.0 / small_scale;
            *sumsq = small_sum_all;
        }
    } else {
        *scale = 1.0;
        *sumsq = medium_sum_all;
    }

    gpu_alloc.deallocate(work, 3);
}

float snrm2(int64_t n, float const *x, int64_t incx) {
    LabeledSection0();

    float result;

    hipblas_catch(hipblasSnrm2_64(gpu::get_blas_handle(), n, x, incx, &result));

    gpu::stream_wait();
    return result;
}

double dnrm2(int64_t n, double const *x, int64_t incx) {
    LabeledSection0();

    double result;

    hipblas_catch(hipblasDnrm2_64(gpu::get_blas_handle(), n, x, incx, &result));

    gpu::stream_wait();
    return result;
}

float scnrm2(int64_t n, std::complex<float> const *x, int64_t incx) {
    LabeledSection0();

    float result;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasScnrm2(gpu::get_blas_handle(), n, (hipblasComplex *)x, incx, &result));
#else
    hipblas_catch(hipblasScnrm2_64(gpu::get_blas_handle(), n, (hipComplex *)x, incx, &result));
#endif

    gpu::stream_wait();
    return result;
}

double dznrm2(int64_t n, std::complex<double> const *x, int64_t incx) {
    LabeledSection0();

    double result;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasDznrm2(gpu::get_blas_handle(), n, (hipblasDoubleComplex *)x, incx, &result));
#else
    hipblas_catch(hipblasDznrm2_64(gpu::get_blas_handle(), n, (hipDoubleComplex *)x, incx, &result));
#endif

    gpu::stream_wait();
    return result;
}

} // namespace einsums::blas::hip