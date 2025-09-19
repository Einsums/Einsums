//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Defines.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <hip/hip_runtime.h>

namespace einsums::blas::hip {

static __global__ void sdirprod_kernel(int n, float *alpha_ptr, float const *x, int incx, float const *y, int incy, float *z, int incz) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int block_size = n / num_threads;
    int remaining  = n % num_threads;

    float alpha = *alpha_ptr;

    if (block_size != 0) {
        float const *curr_x = x + (ptrdiff_t)thread_id * block_size * incx;
        float const *curr_y = y + (ptrdiff_t)thread_id * block_size * incy;
        float       *curr_z = z + (ptrdiff_t)thread_id * block_size * incz;

        for (int i = 0; i < block_size; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }

    if (thread_id == 0) {
        float const *curr_x = x + (ptrdiff_t)num_threads * block_size * incx;
        float const *curr_y = y + (ptrdiff_t)num_threads * block_size * incy;
        float       *curr_z = z + (ptrdiff_t)num_threads * block_size * incz;

        for (int i = 0; i < remaining; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }
}

// Vectorized kernel
static __global__ void sdirprod_all1_kernel(int n, float *alpha, float const *x, float const *y, float *z) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int block_size = n / (4 * num_threads);
    int remaining  = n % (4 * num_threads);

    float4 alpha_vec = make_float4(*alpha, *alpha, *alpha, *alpha);

    if (block_size != 0) {
        float4 const *curr_x = (float4 const *)(x + (ptrdiff_t)thread_id * block_size * 4);
        float4 const *curr_y = (float4 const *)(y + (ptrdiff_t)thread_id * block_size * 4);
        float4       *curr_z = (float4 *)(z + (ptrdiff_t)thread_id * block_size * 4);

        for (int i = 0; i < block_size; i++) {
            curr_z[i] += alpha_vec * curr_x[i] * curr_y[i];
        }
    }

    if (thread_id == 0) {
        float const *curr_x = x + (ptrdiff_t)num_threads * block_size * 4;
        float const *curr_y = y + (ptrdiff_t)num_threads * block_size * 4;
        float       *curr_z = z + (ptrdiff_t)num_threads * block_size * 4;

        for (int i = 0; i < remaining; i++) {
            curr_z[i] += *alpha * curr_x[i] * curr_y[i];
        }
    }
}

static __global__ void ddirprod_kernel(int n, double *alpha_ptr, double const *x, int incx, double const *y, int incy, double *z,
                                       int incz) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int block_size = n / num_threads;
    int remaining  = n % num_threads;

    double alpha = *alpha_ptr;

    if (block_size != 0) {
        double const *curr_x = x + (ptrdiff_t)thread_id * block_size * incx;
        double const *curr_y = y + (ptrdiff_t)thread_id * block_size * incy;
        double       *curr_z = z + (ptrdiff_t)thread_id * block_size * incz;

        for (int i = 0; i < block_size; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }

    if (thread_id == 0) {
        double const *curr_x = x + (ptrdiff_t)num_threads * block_size * incx;
        double const *curr_y = y + (ptrdiff_t)num_threads * block_size * incy;
        double       *curr_z = z + (ptrdiff_t)num_threads * block_size * incz;

        for (int i = 0; i < remaining; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }
}

// Vectorized kernel
static __global__ void ddirprod_all1_kernel(int n, double *alpha, double const *x, double const *y, double *z) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int block_size = n / (4 * num_threads);
    int remaining  = n % (4 * num_threads);

    double4 alpha_vec = make_double4(*alpha, *alpha, *alpha, *alpha);

    if (block_size != 0) {
        double4 const *curr_x = (double4 const *)(x + (ptrdiff_t)thread_id * block_size * 4);
        double4 const *curr_y = (double4 const *)(x + (ptrdiff_t)thread_id * block_size * 4);
        double4       *curr_z = (double4 *)(x + (ptrdiff_t)thread_id * block_size * 4);

        for (int i = 0; i < block_size; i++) {
            curr_z[i] += alpha_vec * curr_x[i] * curr_y[i];
        }
    }

    if (thread_id == 0) {
        double const *curr_x = x + (ptrdiff_t)num_threads * block_size * 4;
        double const *curr_y = y + (ptrdiff_t)num_threads * block_size * 4;
        double       *curr_z = z + (ptrdiff_t)num_threads * block_size * 4;

        for (int i = 0; i < remaining; i++) {
            curr_z[i] += *alpha * curr_x[i] * curr_y[i];
        }
    }
}

static __global__ void cdirprod_kernel(int n, hipFloatComplex *alpha_ptr, hipFloatComplex const *x, int incx, hipFloatComplex const *y,
                                       int incy, hipFloatComplex *z, int incz) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int block_size = n / num_threads;
    int remaining  = n % num_threads;

    hipFloatComplex alpha = *alpha_ptr;

    if (block_size != 0) {
        hipFloatComplex const *curr_x = x + (ptrdiff_t)thread_id * block_size * incx;
        hipFloatComplex const *curr_y = y + (ptrdiff_t)thread_id * block_size * incy;
        hipFloatComplex       *curr_z = z + (ptrdiff_t)thread_id * block_size * incz;

        for (int i = 0; i < block_size; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }

    if (thread_id == 0) {
        hipFloatComplex const *curr_x = x + (ptrdiff_t)num_threads * block_size * incx;
        hipFloatComplex const *curr_y = y + (ptrdiff_t)num_threads * block_size * incy;
        hipFloatComplex       *curr_z = z + (ptrdiff_t)num_threads * block_size * incz;

        for (int i = 0; i < remaining; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }
}

static __global__ void zdirprod_kernel(int n, hipDoubleComplex *alpha_ptr, hipDoubleComplex const *x, int incx, hipDoubleComplex const *y,
                                       int incy, hipDoubleComplex *z, int incz) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int block_size = n / num_threads;
    int remaining  = n % num_threads;

    hipDoubleComplex alpha = *alpha_ptr;

    if (block_size != 0) {
        hipDoubleComplex const *curr_x = x + (ptrdiff_t)thread_id * block_size * incx;
        hipDoubleComplex const *curr_y = y + (ptrdiff_t)thread_id * block_size * incy;
        hipDoubleComplex       *curr_z = z + (ptrdiff_t)thread_id * block_size * incz;

        for (int i = 0; i < block_size; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }

    if (thread_id == 0) {
        hipDoubleComplex const *curr_x = x + (ptrdiff_t)num_threads * block_size * incx;
        hipDoubleComplex const *curr_y = y + (ptrdiff_t)num_threads * block_size * incy;
        hipDoubleComplex       *curr_z = z + (ptrdiff_t)num_threads * block_size * incz;

        for (int i = 0; i < remaining; i++) {
            curr_z[i * incz] += alpha * curr_x[i * incx] * curr_y[i * incy];
        }
    }
}

void sdirprod(int n, float alpha, float const *x, int incx, float const *y, int incy, float *z, int incz) {
    LabeledSection0();

    float *alpha_gpu;

    hip_catch(hipHostRegister((void *)&alpha, sizeof(float), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_gpu, (void *)&alpha, 0));

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks     = gpu::blocks(n / 4);
        auto block_size = gpu::block_size(n / 4);

        sdirprod_all1_kernel<<<blocks, block_size, 0, gpu::get_stream()>>>(n, alpha_gpu, x, y, z);
    } else {
        auto blocks     = gpu::blocks(n);
        auto block_size = gpu::block_size(n);

        sdirprod_kernel<<<blocks, block_size, 0, gpu::get_stream()>>>(n, alpha_gpu, x, incx, y, incy, z, incz);
    }

    gpu::stream_wait();

    hip_catch(hipHostUnregister((void *)&alpha));
}

void ddirprod(int n, double alpha, double const *x, int incx, double const *y, int incy, double *z, int incz) {
    LabeledSection0();

    double *alpha_gpu;

    hip_catch(hipHostRegister((void *)&alpha, sizeof(double), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_gpu, (void *)&alpha, 0));

    if (incx == 1 && incy == 1 && incz == 1) {
        auto blocks     = gpu::blocks(n / 4);
        auto block_size = gpu::block_size(n / 4);

        ddirprod_all1_kernel<<<blocks, block_size, 0, gpu::get_stream()>>>(n, alpha_gpu, x, y, z);
    } else {
        auto blocks     = gpu::blocks(n);
        auto block_size = gpu::block_size(n);

        ddirprod_kernel<<<blocks, block_size, 0, gpu::get_stream()>>>(n, alpha_gpu, x, incx, y, incy, z, incz);
    }

    gpu::stream_wait();

    hip_catch(hipHostUnregister((void *)&alpha));
}

void cdirprod(int n, std::complex<float> alpha, std::complex<float> const *x, int incx, std::complex<float> const *y, int incy,
              std::complex<float> *z, int incz) {
    LabeledSection0();

    hipComplex *alpha_gpu;

    hip_catch(hipHostRegister((void *)&alpha, sizeof(hipComplex), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_gpu, (void *)&alpha, 0));

    auto blocks     = gpu::blocks(n);
    auto block_size = gpu::block_size(n);

    cdirprod_kernel<<<blocks, block_size, 0, gpu::get_stream()>>>(n, alpha_gpu, (hipComplex const *)x, incx, (hipComplex const *)y, incy,
                                                                  (hipComplex *)z, incz);

    gpu::stream_wait();

    hip_catch(hipHostUnregister((void *)&alpha));
}

void zdirprod(int n, std::complex<double> alpha, std::complex<double> const *x, int incx, std::complex<double> const *y, int incy,
              std::complex<double> *z, int incz) {
    LabeledSection0();

    hipDoubleComplex *alpha_gpu;

    hip_catch(hipHostRegister((void *)&alpha, sizeof(hipDoubleComplex), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&alpha_gpu, (void *)&alpha, 0));

    auto blocks     = gpu::blocks(n);
    auto block_size = gpu::block_size(n);

    zdirprod_kernel<<<blocks, block_size, 0, gpu::get_stream()>>>(n, alpha_gpu, (hipDoubleComplex const *)x, incx,
                                                                  (hipDoubleComplex const *)y, incy, (hipDoubleComplex *)z, incz);

    gpu::stream_wait();

    hip_catch(hipHostUnregister((void *)&alpha));
}

} // namespace einsums::blas::hip