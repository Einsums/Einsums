//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hipblas/hipblas.h>

#include "Common.hpp"

namespace einsums::blas::hip {
float sasum(int64_t n, float const *x, int64_t incx) {
    LabeledSection0();

    float out;

    hipblas_catch(hipblasSasum_64(gpu::get_blas_handle(), n, x, incx, &out));
    gpu::stream_wait();

    return out;
}

double dasum(int64_t n, double const *x, int64_t incx) {
    LabeledSection0();

    double out;

    hipblas_catch(hipblasDasum_64(gpu::get_blas_handle(), n, x, incx, &out));
    gpu::stream_wait();

    return out;
}

float scasum(int64_t n, std::complex<float> const *x, int64_t incx) {
    LabeledSection0();

    float out;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasScasum_64(gpu::get_blas_handle(), n, (hipblasComplex const *)x, incx, &out));
#else
    hipblas_catch(hipblasScasum_64(gpu::get_blas_handle(), n, (hipComplex const *)x, incx, &out));
#endif
    gpu::stream_wait();

    return out;
}

double dzasum(int64_t n, std::complex<double> const *x, int64_t incx) {
    LabeledSection0();

    double out;

#if hipblasVersionMajor < 3
    hipblas_catch(hipblasDzasum_64(gpu::get_blas_handle(), n, (hipblasDoubleComplex const *)x, incx, &out));
#else
    hipblas_catch(hipblasDzasum_64(gpu::get_blas_handle(), n, (hipDoubleComplex const *)x, incx, &out));
#endif
    gpu::stream_wait();

    return out;
}

__global__ void scsum1_kernel(int64_t n, hipComplex const *x, int64_t incx, float *result) {
    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;

    ptrdiff_t const mega_block_size = n / (num_threads);
    int64_t const       remainder       = n % (num_threads);

    if (mega_block_size != 0) {
        hipComplex const *current_block = x + (ptrdiff_t)incx * thread_id * mega_block_size;

        // Loop through the elements in the block.
        float work_var{0.0};

        // Do the actual thing.
        for (ptrdiff_t i = 0; i < mega_block_size; i++) {
            hipComplex curr_val = current_block[i * incx];
            work_var += hypotf(curr_val.x, curr_val.y);
        }

        // Poll the threads in the warp to combine their work.
        // #pragma unroll
        //         for (int i = warpSize / 2; i >= 1; i /= 2) {
        //             work_var += __shfl_down(work_var, i);
        //         }

        //         if (inwarp_id == 0) {
        atomicAdd(result, work_var);
        // }
    }

    if (thread_id == 0) {
        hipComplex const *remaining_values = x + (ptrdiff_t)incx * num_threads * mega_block_size;

        for (int i = 0; i < remainder; i++) {
            hipComplex curr_val = remaining_values[i * incx];
            *result += hypotf(curr_val.x, curr_val.y);
        }
    }
}

__global__ void dzsum1_kernel(int64_t n, hipDoubleComplex const *x, int64_t incx, double *result) {
    int thread_id, num_threads;
    get_worker_info(thread_id, num_threads);

    int inwarp_id = thread_id % warpSize;

    ptrdiff_t const mega_block_size = n / (num_threads);
    int64_t const       remainder       = n % (num_threads);

    if (mega_block_size != 0) {
        hipDoubleComplex const *current_block = x + (ptrdiff_t)incx * thread_id * mega_block_size;

        // Loop through the elements in the block.
        double work_var{0.0};

        // Do the actual thing.
        for (ptrdiff_t i = 0; i < mega_block_size; i++) {
            hipDoubleComplex curr_val = current_block[i * incx];
            work_var += hypot(curr_val.x, curr_val.y);
        }

        // Poll the threads in the warp to combine their work.
        // #pragma unroll
        //         for (int i = warpSize / 2; i >= 1; i /= 2) {
        //             work_var += __shfl_down(work_var, i);
        //         }

        //         if (inwarp_id == 0) {
        atomicAdd(result, work_var);
        // }
    }

    if (thread_id == 0) {
        hipDoubleComplex const *remaining_values = x + (ptrdiff_t)incx * num_threads * mega_block_size;

        for (int i = 0; i < remainder; i++) {
            hipDoubleComplex curr_val = remaining_values[i * incx];
            *result += hypotf(curr_val.x, curr_val.y);
        }
    }
}

float scsum1(int64_t n, std::complex<float> const *x, int64_t incx) {
    LabeledSection0();

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    float out = 0.0;

    hip_catch(hipHostRegister((void *)&out, sizeof(float), hipHostRegisterDefault));

    float *gpu_out;

    hip_catch(hipHostGetDevicePointer((void **)&gpu_out, (void *)&out, 0));

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    scsum1_kernel<<<block_size, blocks, 0, stream>>>(n, (hipComplex const *)x, incx, gpu_out);
    gpu::stream_wait();

    hip_catch(hipHostUnregister(&out));

    return out;
}

double dzsum1(int64_t n, std::complex<double> const *x, int64_t incx) {
    LabeledSection0();

    auto block_size = gpu::block_size(n);
    auto blocks     = gpu::blocks(n);

    double out = 0.0;

    hip_catch(hipHostRegister((void *)&out, sizeof(double), hipHostRegisterDefault));

    double *gpu_out;

    hip_catch(hipHostGetDevicePointer((void **)&gpu_out, (void *)&out, 0));

    hipStream_t stream;

    hipblas_catch(hipblasGetStream(gpu::get_blas_handle(), &stream));

    dzsum1_kernel<<<block_size, blocks, 0, stream>>>(n, (hipDoubleComplex const *)x, incx, gpu_out);
    gpu::stream_wait();

    hip_catch(hipHostUnregister(&out));

    return out;
}

} // namespace einsums::blas::hip