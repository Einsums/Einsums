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

#include <exception>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {

__global__ void slassq_kernel(int n, float const *x, int incx, float *scale, float *sumsq, float *big_sum_all, float *medium_sum_all,
                              float *small_sum_all) {
    if (isnan(*scale) || isnan(*sumsq)) {
        return;
    }

    constexpr float sfmin  = std::numeric_limits<float>::min();
    constexpr float small  = 1.0f / std::numeric_limits<float>::max();
    constexpr float smlnum = (small > sfmin) ? small * (1.0f + std::numeric_limits<float>::epsilon()) : sfmin;
    constexpr float bignum = 1.0f / smlnum;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;

    int parallel_elems  = n / num_threads;
    int remaining_elems = n % num_threads;

    bool  not_big    = true;
    float small_sum  = 0.0f;
    float medium_sum = 0.0f;
    float big_sum    = 0.0f;

    float const *x_curr = x + (ptrdiff_t)thread_id * parallel_elems * incx;

    // Do the things.
    for (int i = 0; i < parallel_elems; i += incx) {
        float ax = fabsf(x_curr[i]);
        if (ax > bignum) {
            big_sum += (ax * smlnum) * (ax * smlnum);
            not_big = false;
        } else if (ax < smlnum) {
            if (not_big) {
                small_sum += (ax * bignum) * (ax * bignum);
            }
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
#pragma unroll
    for (int i = warpSize / 2; i >= 1; i /= 2) {
        small_sum += __shfl_down(small_sum, i);
        medium_sum += __shfl_down(medium_sum, i);
        big_sum += __shfl_down(big_sum, i);
    }

    // Handle the remaining.
    if (thread_id == 0) {
        // Consider the scale and sum from the input.
        if (*sumsq > 0.0f) {
            float ax = *scale * sqrtf(*sumsq);
            if (ax > bignum) {
                if (*scale > 1.0f) {
                    *scale *= smlnum;
                    big_sum += *scale * *scale * *sumsq;
                } else {
                    big_sum += *scale * *scale * smlnum * smlnum * *sumsq;
                }
            } else if (ax < smlnum) {
                if (*scale < 1.0f) {
                    *scale *= bignum;
                    small_sum += *scale * *scale * *sumsq;
                } else {
                    small_sum += *scale * *scale * bignum * bignum * *sumsq;
                }
            } else {
                medium_sum += *scale * *scale * *sumsq;
            }
        }

        // Collect the unprocessed values.
        float const *x_end = x + (ptrdiff_t)incx * num_threads * parallel_elems;
        for (int i = 0; i < remaining_elems; i += incx) {
            float ax = fabsf(x_curr[i]);
            if (ax > bignum) {
                big_sum += (ax * smlnum) * (ax * smlnum);
                not_big = false;
            } else if (ax < smlnum) {
                if (not_big) {
                    small_sum += (ax * bignum) * (ax * bignum);
                }
            } else {
                medium_sum += ax * ax;
            }
        }
    }

    // Combine the sums.
    if (inwarp_id == 0) {
        atomicAdd(big_sum_all, big_sum);
        atomicAdd(medium_sum_all, medium_sum);
        atomicAdd(small_sum_all, small_sum);
    }
    __syncthreads();

    // Finally, set the scale and sum.
    if (thread_id == 0) {
        if (*big_sum_all > 0.0f) {
            if (*medium_sum_all > 0.0f || isnan(*medium_sum_all)) {
                *big_sum_all += (*medium_sum_all * smlnum) * smlnum;
            }
            *scale = 1.0f / smlnum;
            *sumsq = *big_sum_all;
        } else if (*small_sum_all > 0.0f) {
            float ymin, ymax;
            if (*medium_sum_all > 0.0f || isnan(*medium_sum_all)) {
                *medium_sum_all = sqrtf(*medium_sum_all);
                *small_sum_all  = sqrtf(*small_sum_all) / bignum;
                if (*small_sum_all > *medium_sum_all) {
                    ymin = *medium_sum_all;
                    ymax = *small_sum_all;
                } else {
                    ymin = *small_sum_all;
                    ymax = *medium_sum_all;
                }

                *scale = 1.0f;
                *sumsq = ymax * ymax * (1.0f + (ymin / ymax) * (ymin / ymax));
            } else {
                *scale = 1.0f / bignum;
                *sumsq = *small_sum_all;
            }
        } else {
            *scale = 1.0f;
            *sumsq = *medium_sum_all;
        }
    }
}

__global__ void dlassq_kernel(int n, double const *x, int incx, double *scale, double *sumsq, double *big_sum_all, double *medium_sum_all,
                              double *small_sum_all) {
    if (isnan(*scale) || isnan(*sumsq)) {
        return;
    }

    constexpr double sfmin  = std::numeric_limits<double>::min();
    constexpr double small  = 1.0 / std::numeric_limits<double>::max();
    constexpr double smlnum = (small > sfmin) ? small * (1.0 + std::numeric_limits<double>::epsilon()) : sfmin;
    constexpr double bignum = 1.0 / smlnum;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;

    int parallel_elems  = n / num_threads;
    int remaining_elems = n % num_threads;

    bool   not_big    = true;
    double small_sum  = 0.0;
    double medium_sum = 0.0;
    double big_sum    = 0.0;

    double const *x_curr = x + (ptrdiff_t)thread_id * parallel_elems * incx;

    // Do the things.
    for (int i = 0; i < parallel_elems; i += incx) {
        double ax = fabs(x_curr[i]);
        if (ax > bignum) {
            big_sum += (ax * smlnum) * (ax * smlnum);
            not_big = false;
        } else if (ax < smlnum) {
            if (not_big) {
                small_sum += (ax * bignum) * (ax * bignum);
            }
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
#pragma unroll
    for (int i = warpSize / 2; i >= 1; i /= 2) {
        small_sum += __shfl_down(small_sum, i);
        medium_sum += __shfl_down(medium_sum, i);
        big_sum += __shfl_down(big_sum, i);
    }

    // Handle the remaining.
    if (thread_id == 0) {
        // Consider the scale and sum from the input.
        if (*sumsq > 0.0) {
            double ax = *scale * sqrt(*sumsq);
            if (ax > bignum) {
                if (*scale > 1.0) {
                    *scale *= smlnum;
                    big_sum += *scale * *scale * *sumsq;
                } else {
                    big_sum += *scale * *scale * smlnum * smlnum * *sumsq;
                }
            } else if (ax < smlnum) {
                if (*scale < 1.0) {
                    *scale *= bignum;
                    small_sum += *scale * *scale * *sumsq;
                } else {
                    small_sum += *scale * *scale * bignum * bignum * *sumsq;
                }
            } else {
                medium_sum += *scale * *scale * *sumsq;
            }
        }

        // Collect the unprocessed values.
        double const *x_end = x + (ptrdiff_t)incx * num_threads * parallel_elems;
        for (int i = 0; i < remaining_elems; i += incx) {
            double ax = fabs(x_curr[i]);
            if (ax > bignum) {
                big_sum += (ax * smlnum) * (ax * smlnum);
                not_big = false;
            } else if (ax < smlnum) {
                if (not_big) {
                    small_sum += (ax * bignum) * (ax * bignum);
                }
            } else {
                medium_sum += ax * ax;
            }
        }
    }

    // Combine the sums.
    if (inwarp_id == 0) {
        atomicAdd(big_sum_all, big_sum);
        atomicAdd(medium_sum_all, medium_sum);
        atomicAdd(small_sum_all, small_sum);
    }
    __syncthreads();

    // Finally, set the scale and sum.
    if (thread_id == 0) {
        if (*big_sum_all > 0.0) {
            if (*medium_sum_all > 0.0 || isnan(*medium_sum_all)) {
                *big_sum_all += (*medium_sum_all * smlnum) * smlnum;
            }
            *scale = 1.0 / smlnum;
            *sumsq = *big_sum_all;
        } else if (*small_sum_all > 0.0) {
            double ymin, ymax;
            if (*medium_sum_all > 0.0 || isnan(*medium_sum_all)) {
                *medium_sum_all = sqrt(*medium_sum_all);
                *small_sum_all  = sqrt(*small_sum_all) / bignum;
                if (*small_sum_all > *medium_sum_all) {
                    ymin = *medium_sum_all;
                    ymax = *small_sum_all;
                } else {
                    ymin = *small_sum_all;
                    ymax = *medium_sum_all;
                }

                *scale = 1.0;
                *sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
            } else {
                *scale = 1.0 / bignum;
                *sumsq = *small_sum_all;
            }
        } else {
            *scale = 1.0;
            *sumsq = *medium_sum_all;
        }
    }
}

__global__ void classq_kernel(int n, hipComplex const *x, int incx, float *scale, float *sumsq, float *big_sum_all, float *medium_sum_all,
                              float *small_sum_all) {
    if (isnan(*scale) || isnan(*sumsq)) {
        return;
    }

    constexpr float sfmin  = std::numeric_limits<float>::min();
    constexpr float small  = 1.0f / std::numeric_limits<float>::max();
    constexpr float smlnum = (small > sfmin) ? small * (1.0f + std::numeric_limits<float>::epsilon()) : sfmin;
    constexpr float bignum = 1.0f / smlnum;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;

    int parallel_elems  = n / num_threads;
    int remaining_elems = n % num_threads;

    bool  not_big    = true;
    float small_sum  = 0.0f;
    float medium_sum = 0.0f;
    float big_sum    = 0.0f;

    hipComplex const *x_curr = x + (ptrdiff_t)thread_id * parallel_elems * incx;

    // Do the things.
    for (int i = 0; i < parallel_elems; i += incx) {
        float ax = fabsf(x_curr[i].x);
        if (ax > bignum) {
            big_sum += (ax * smlnum) * (ax * smlnum);
            not_big = false;
        } else if (ax < smlnum) {
            if (not_big) {
                small_sum += (ax * bignum) * (ax * bignum);
            }
        } else {
            medium_sum += ax * ax;
        }

        ax = fabsf(x_curr[i].y);
        if (ax > bignum) {
            big_sum += (ax * smlnum) * (ax * smlnum);
            not_big = false;
        } else if (ax < smlnum) {
            if (not_big) {
                small_sum += (ax * bignum) * (ax * bignum);
            }
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
#pragma unroll
    for (int i = warpSize / 2; i >= 1; i /= 2) {
        small_sum += __shfl_down(small_sum, i);
        medium_sum += __shfl_down(medium_sum, i);
        big_sum += __shfl_down(big_sum, i);
    }

    // Handle the remaining.
    if (thread_id == 0) {
        // Consider the scale and sum from the input.
        if (*sumsq > 0.0f) {
            float ax = *scale * sqrtf(*sumsq);
            if (ax > bignum) {
                if (*scale > 1.0f) {
                    *scale *= smlnum;
                    big_sum += *scale * *scale * *sumsq;
                } else {
                    big_sum += *scale * *scale * smlnum * smlnum * *sumsq;
                }
            } else if (ax < smlnum) {
                if (*scale < 1.0f) {
                    *scale *= bignum;
                    small_sum += *scale * *scale * *sumsq;
                } else {
                    small_sum += *scale * *scale * bignum * bignum * *sumsq;
                }
            } else {
                medium_sum += *scale * *scale * *sumsq;
            }
        }

        // Collect the unprocessed values.
        hipComplex const *x_end = x + (ptrdiff_t)incx * num_threads * parallel_elems;
        for (int i = 0; i < remaining_elems; i += incx) {
            float ax = fabsf(x_curr[i].x);
            if (ax > bignum) {
                big_sum += (ax * smlnum) * (ax * smlnum);
                not_big = false;
            } else if (ax < smlnum) {
                if (not_big) {
                    small_sum += (ax * bignum) * (ax * bignum);
                }
            } else {
                medium_sum += ax * ax;
            }

            ax = fabsf(x_curr[i].y);
            if (ax > bignum) {
                big_sum += (ax * smlnum) * (ax * smlnum);
                not_big = false;
            } else if (ax < smlnum) {
                if (not_big) {
                    small_sum += (ax * bignum) * (ax * bignum);
                }
            } else {
                medium_sum += ax * ax;
            }
        }
    }

    // Combine the sums.
    if (inwarp_id == 0) {
        atomicAdd(big_sum_all, big_sum);
        atomicAdd(medium_sum_all, medium_sum);
        atomicAdd(small_sum_all, small_sum);
    }
    __syncthreads();

    // Finally, set the scale and sum.
    if (thread_id == 0) {
        if (*big_sum_all > 0.0f) {
            if (*medium_sum_all > 0.0f || isnan(*medium_sum_all)) {
                *big_sum_all += (*medium_sum_all * smlnum) * smlnum;
            }
            *scale = 1.0f / smlnum;
            *sumsq = *big_sum_all;
        } else if (*small_sum_all > 0.0f) {
            float ymin, ymax;
            if (*medium_sum_all > 0.0f || isnan(*medium_sum_all)) {
                *medium_sum_all = sqrtf(*medium_sum_all);
                *small_sum_all  = sqrtf(*small_sum_all) / bignum;
                if (*small_sum_all > *medium_sum_all) {
                    ymin = *medium_sum_all;
                    ymax = *small_sum_all;
                } else {
                    ymin = *small_sum_all;
                    ymax = *medium_sum_all;
                }

                *scale = 1.0f;
                *sumsq = ymax * ymax * (1.0f + (ymin / ymax) * (ymin / ymax));
            } else {
                *scale = 1.0f / bignum;
                *sumsq = *small_sum_all;
            }
        } else {
            *scale = 1.0f;
            *sumsq = *medium_sum_all;
        }
    }
}

__global__ void zlassq_kernel(int n, hipDoubleComplex const *x, int incx, double *scale, double *sumsq, double *big_sum_all,
                              double *medium_sum_all, double *small_sum_all) {
    if (isnan(*scale) || isnan(*sumsq)) {
        return;
    }

    constexpr double sfmin  = std::numeric_limits<double>::min();
    constexpr double small  = 1.0 / std::numeric_limits<double>::max();
    constexpr double smlnum = (small > sfmin) ? small * (1.0 + std::numeric_limits<double>::epsilon()) : sfmin;
    constexpr double bignum = 1.0 / smlnum;

    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;

    int parallel_elems  = n / num_threads;
    int remaining_elems = n % num_threads;

    bool   not_big    = true;
    double small_sum  = 0.0;
    double medium_sum = 0.0;
    double big_sum    = 0.0;

    hipDoubleComplex const *x_curr = x + (ptrdiff_t)thread_id * parallel_elems * incx;

    // Do the things.
    for (int i = 0; i < parallel_elems; i += incx) {
        double ax = fabs(x_curr[i].x);
        if (ax > bignum) {
            big_sum += (ax * smlnum) * (ax * smlnum);
            not_big = false;
        } else if (ax < smlnum) {
            if (not_big) {
                small_sum += (ax * bignum) * (ax * bignum);
            }
        } else {
            medium_sum += ax * ax;
        }

        ax = fabs(x_curr[i].y);
        if (ax > bignum) {
            big_sum += (ax * smlnum) * (ax * smlnum);
            not_big = false;
        } else if (ax < smlnum) {
            if (not_big) {
                small_sum += (ax * bignum) * (ax * bignum);
            }
        } else {
            medium_sum += ax * ax;
        }
    }

    // Combine accross warps.
#pragma unroll
    for (int i = warpSize / 2; i >= 1; i /= 2) {
        small_sum += __shfl_down(small_sum, i);
        medium_sum += __shfl_down(medium_sum, i);
        big_sum += __shfl_down(big_sum, i);
    }

    // Handle the remaining.
    if (thread_id == 0) {
        // Consider the scale and sum from the input.
        if (*sumsq > 0.0) {
            double ax = *scale * sqrt(*sumsq);
            if (ax > bignum) {
                if (*scale > 1.0) {
                    *scale *= smlnum;
                    big_sum += *scale * *scale * *sumsq;
                } else {
                    big_sum += *scale * *scale * smlnum * smlnum * *sumsq;
                }
            } else if (ax < smlnum) {
                if (*scale < 1.0) {
                    *scale *= bignum;
                    small_sum += *scale * *scale * *sumsq;
                } else {
                    small_sum += *scale * *scale * bignum * bignum * *sumsq;
                }
            } else {
                medium_sum += *scale * *scale * *sumsq;
            }
        }

        // Collect the unprocessed values.
        hipDoubleComplex const *x_end = x + (ptrdiff_t)incx * num_threads * parallel_elems;
        for (int i = 0; i < remaining_elems; i += incx) {
            double ax = fabs(x_curr[i].x);
            if (ax > bignum) {
                big_sum += (ax * smlnum) * (ax * smlnum);
                not_big = false;
            } else if (ax < smlnum) {
                if (not_big) {
                    small_sum += (ax * bignum) * (ax * bignum);
                }
            } else {
                medium_sum += ax * ax;
            }

            ax = fabs(x_curr[i].y);
            if (ax > bignum) {
                big_sum += (ax * smlnum) * (ax * smlnum);
                not_big = false;
            } else if (ax < smlnum) {
                if (not_big) {
                    small_sum += (ax * bignum) * (ax * bignum);
                }
            } else {
                medium_sum += ax * ax;
            }
        }
    }

    // Combine the sums.
    if (inwarp_id == 0) {
        atomicAdd(big_sum_all, big_sum);
        atomicAdd(medium_sum_all, medium_sum);
        atomicAdd(small_sum_all, small_sum);
    }
    __syncthreads();

    // Finally, set the scale and sum.
    if (thread_id == 0) {
        if (*big_sum_all > 0.0) {
            if (*medium_sum_all > 0.0 || isnan(*medium_sum_all)) {
                *big_sum_all += (*medium_sum_all * smlnum) * smlnum;
            }
            *scale = 1.0 / smlnum;
            *sumsq = *big_sum_all;
        } else if (*small_sum_all > 0.0) {
            double ymin, ymax;
            if (*medium_sum_all > 0.0 || isnan(*medium_sum_all)) {
                *medium_sum_all = sqrt(*medium_sum_all);
                *small_sum_all  = sqrt(*small_sum_all) / bignum;
                if (*small_sum_all > *medium_sum_all) {
                    ymin = *medium_sum_all;
                    ymax = *small_sum_all;
                } else {
                    ymin = *small_sum_all;
                    ymax = *medium_sum_all;
                }

                *scale = 1.0;
                *sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
            } else {
                *scale = 1.0 / bignum;
                *sumsq = *small_sum_all;
            }
        } else {
            *scale = 1.0;
            *sumsq = *medium_sum_all;
        }
    }
}

void slassq(int n, float const *x, int incx, float *scale, float *sumsq) {
    LabeledSection("slassq");

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<float>();

    auto work = gpu_alloc.allocate(3);

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    slassq_kernel<<<block_size, blocks, 0, stream>>>(n, x, incx, scale, sumsq, work, work + 1, work + 2);
    gpu::stream_wait();

    gpu_alloc.deallocate(work, 3);
}

void dlassq(int n, double const *x, int incx, double *scale, double *sumsq) {
    LabeledSection("dlassq");

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<double>();

    auto work = gpu_alloc.allocate(3);

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    dlassq_kernel<<<block_size, blocks, 0, stream>>>(n, x, incx, scale, sumsq, work, work + 1, work + 2);
    gpu::stream_wait();

    gpu_alloc.deallocate(work, 3);
}

void classq(int n, std::complex<float> const *x, int incx, float *scale, float *sumsq) {
    LabeledSection("classq");

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<float>();

    auto work = gpu_alloc.allocate(3);

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    classq_kernel<<<block_size, blocks, 0, stream>>>(n, (hipComplex *)x, incx, scale, sumsq, work, work + 1, work + 2);
    gpu::stream_wait();

    gpu_alloc.deallocate(work, 3);
}

void zlassq(int n, std::complex<double> const *x, int incx, double *scale, double *sumsq) {
    LabeledSection0();

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    auto gpu_alloc = gpu::GPUAllocator<double>();

    auto work = gpu_alloc.allocate(3);

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    zlassq_kernel<<<block_size, blocks, 0, stream>>>(n, (hipDoubleComplex *)x, incx, scale, sumsq, work, work + 1, work + 2);
    gpu::stream_wait();

    gpu_alloc.deallocate(work, 3);
}

float snrm2(int n, float const *x, int incx) {
    LabeledSection0();

    float result;

    hipblas_catch(hipblasSnrm2(gpu::get_blas_handle(), n, x, incx, &result));

    gpu::stream_wait();
    return result;
}

double dnrm2(int n, double const *x, int incx) {
    LabeledSection0();

    double result;

    hipblas_catch(hipblasDnrm2(gpu::get_blas_handle(), n, x, incx, &result));

    gpu::stream_wait();
    return result;
}

float scnrm2(int n, std::complex<float> const *x, int incx) {
    LabeledSection0();

    float result;

    hipblas_catch(hipblasScnrm2(gpu::get_blas_handle(), n, (hipblasComplex *)x, incx, &result));

    gpu::stream_wait();
    return result;
}

double dznrm2(int n, std::complex<double> const *x, int incx) {
    LabeledSection0();

    double result;

    hipblas_catch(hipblasDznrm2(gpu::get_blas_handle(), n, (hipblasDoubleComplex *)x, incx, &result));

    gpu::stream_wait();
    return result;
}

} // namespace einsums::blas::hip