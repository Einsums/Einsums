//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <limits>

#include "Common.hpp"

namespace einsums::blas::hip {

void sscal(int n, float alpha, float *vec, int inc) {
    LabeledSection0();

    float *alpha_gpu = gpu::register_host_variable(alpha);

    hipblas_catch(hipblasSscal(gpu::get_blas_handle(), n, alpha_gpu, vec, inc));

    gpu::unregister_host_variable(alpha);
}

void dscal(int n, double alpha, double *vec, int inc) {
    LabeledSection0();

    double *alpha_gpu = gpu::register_host_variable(alpha);

    hipblas_catch(hipblasDscal(gpu::get_blas_handle(), n, alpha_gpu, vec, inc));

    gpu::unregister_host_variable(alpha);
}

void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc) {
    LabeledSection0();

    hipblasComplex *alpha_gpu = (hipblasComplex *)gpu::register_host_variable(alpha);

    hipblas_catch(hipblasCscal(gpu::get_blas_handle(), n, alpha_gpu, (hipblasComplex *)vec, inc));

    gpu::unregister_host_variable(alpha);
}

void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc) {
    LabeledSection0();

    hipblasDoubleComplex *alpha_gpu = (hipblasDoubleComplex *)gpu::register_host_variable(alpha);

    hipblas_catch(hipblasZscal(gpu::get_blas_handle(), n, alpha_gpu, (hipblasDoubleComplex *)vec, inc));

    gpu::unregister_host_variable(alpha);
}

void csscal(int n, float alpha, std::complex<float> *vec, int inc) {
    LabeledSection0();

    float *alpha_gpu = gpu::register_host_variable(alpha);

    hipblas_catch(hipblasCsscal(gpu::get_blas_handle(), n, alpha_gpu, (hipblasComplex *)vec, inc));

    gpu::unregister_host_variable(alpha);
}

void zdscal(int n, double alpha, std::complex<double> *vec, int inc) {
    LabeledSection0();

    double *alpha_gpu = gpu::register_host_variable(alpha);

    hipblas_catch(hipblasZdscal(gpu::get_blas_handle(), n, alpha_gpu, (hipblasDoubleComplex *)vec, inc));

    gpu::unregister_host_variable(alpha);
}

void srscl(int n, float alpha, float *vec, int inc) {
    LabeledSection0();

    if (n <= 0) {
        return;
    }

    constexpr float sfmin  = std::numeric_limits<float>::min();
    constexpr float small  = 1.0f / std::numeric_limits<float>::max();
    constexpr float smlnum = (small > sfmin) ? small * (1.0f + std::numeric_limits<float>::epsilon()) : sfmin;
    constexpr float bignum = 1.0f / smlnum;

    float den = alpha;
    float num = 1.0f;
    float next_scale;

    bool finished;

    do {
        float den1 = den * smlnum;
        float num1 = num / bignum;

        if (std::abs(den1) > std::abs(num) && num != 0.0f) {
            next_scale = smlnum;
            finished   = false;
            den        = den1;
        } else if (std::abs(num1) > std::abs(den)) {
            next_scale = bignum;
            finished   = false;
            num        = num1;
        } else {
            next_scale = num / den;
            finished   = true;
        }

        sscal(n, next_scale, vec, inc);
    } while (!finished);
}

void drscl(int n, double alpha, double *vec, int inc) {
    LabeledSection0();

    if (n <= 0) {
        return;
    }

    constexpr double sfmin  = std::numeric_limits<double>::min();
    constexpr double small  = 1.0 / std::numeric_limits<double>::max();
    constexpr double smlnum = (small > sfmin) ? small * (1.0 + std::numeric_limits<double>::epsilon()) : sfmin;
    constexpr double bignum = 1.0 / smlnum;

    double den = alpha;
    double num = 1.0;
    double next_scale;

    bool finished;

    do {
        double den1 = den * smlnum;
        double num1 = num / bignum;

        if (std::abs(den1) > std::abs(num) && num != 0.0) {
            next_scale = smlnum;
            finished   = false;
            den        = den1;
        } else if (std::abs(num1) > std::abs(den)) {
            next_scale = bignum;
            finished   = false;
            num        = num1;
        } else {
            next_scale = num / den;
            finished   = true;
        }

        dscal(n, next_scale, vec, inc);
    } while (!finished);
}

void csrscl(int n, float alpha, std::complex<float> *vec, int inc) {
    LabeledSection0();

    if (n <= 0) {
        return;
    }

    constexpr float sfmin  = std::numeric_limits<float>::min();
    constexpr float small  = 1.0f / std::numeric_limits<float>::max();
    constexpr float smlnum = (small > sfmin) ? small * (1.0f + std::numeric_limits<float>::epsilon()) : sfmin;
    constexpr float bignum = 1.0f / smlnum;

    float den = alpha;
    float num = 1.0f;
    float next_scale;

    bool finished;

    do {
        float den1 = den * smlnum;
        float num1 = num / bignum;

        if (std::abs(den1) > std::abs(num) && num != 0.0f) {
            next_scale = smlnum;
            finished   = false;
            den        = den1;
        } else if (std::abs(num1) > std::abs(den)) {
            next_scale = bignum;
            finished   = false;
            num        = num1;
        } else {
            next_scale = num / den;
            finished   = true;
        }

        csscal(n, next_scale, vec, inc);
    } while (!finished);
}

void zdrscl(int n, double alpha, std::complex<double> *vec, int inc) {
    LabeledSection0();

    if (n <= 0) {
        return;
    }

    constexpr double sfmin  = std::numeric_limits<double>::min();
    constexpr double small  = 1.0 / std::numeric_limits<double>::max();
    constexpr double smlnum = (small > sfmin) ? small * (1.0 + std::numeric_limits<double>::epsilon()) : sfmin;
    constexpr double bignum = 1.0 / smlnum;

    double den = alpha;
    double num = 1.0;
    double next_scale;

    bool finished;

    do {
        double den1 = den * smlnum;
        double num1 = num / bignum;

        if (std::abs(den1) > std::abs(num) && num != 0.0) {
            next_scale = smlnum;
            finished   = false;
            den        = den1;
        } else if (std::abs(num1) > std::abs(den)) {
            next_scale = bignum;
            finished   = false;
            num        = num1;
        } else {
            next_scale = num / den;
            finished   = true;
        }

        zdscal(n, next_scale, vec, inc);
    } while (!finished);
}

__global__ void slascl_full_matrix(float mult, int m, int n, float *matrix, int lda) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    // Vectorization for the win!
    float4 mult_vec = make_float4(mult, mult, mult, mult);

    // Find out where the thread is in the block.
    int thread_x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_rows    = blockDim.y * gridDim.y;
    int thread_cols    = blockDim.x * gridDim.x;

    // Figure out how many elements each thread can handle.
    int work_rows = m / (4 * thread_rows);
    int work_cols = n / thread_cols;

    // If there are some remaining, figure out how many.
    int remaining_rows = m % (4 * thread_rows);
    int remaining_cols = n % thread_cols;

    // Find the number of extra vectorable columns and unvectorable columns.
    int extra_vec_rows = remaining_rows / 4;
    int unvec_rows     = remaining_rows % 4;

    int x_coord = thread_x_coord * work_rows;
    int y_coord = thread_y_coord * work_cols;

    // If the thread's row is less than the number of remaining rows, then assign it the work of one of the remaining rows.
    if (thread_y_coord < extra_vec_rows) {
        work_rows++;
        y_coord++;
    } else {
        y_coord += extra_vec_rows;
    }

    if (thread_x_coord < remaining_cols) {
        work_cols++;
        x_coord++;
    } else {
        x_coord += remaining_cols;
    }

    // If the thread does not have any work, exit.
    if (work_rows == 0 || work_cols == 0) {
        return;
    }

    if (x_coord >= n || y_coord >= m) {
        return;
    }

    // If the thread's

    // Now, scale.

    // Start with the vectorable rows and columns.
    for (int i = 0; i < work_cols; i++) {
        float4 *curr_work_pos = (float4 *)matrix + (ptrdiff_t)(x_coord + i) * lda + (ptrdiff_t)y_coord;
#ifdef EINSUMS_DEBUG
        assert(x_coord + i < n);
#endif

        for (int j = 0; j < work_rows; j++) {
#ifdef EINSUMS_DEBUG
            assert(y_coord + j < m);
#endif
            curr_work_pos[j] *= mult_vec;
        }
    }

    // Then do the unvectorable columns.
    if (threadIdx.x == 0) {
        int row_offset = m - (m % 4);
        for (int i = 0; i < work_cols; i++) {
#ifdef EINSUMS_DEBUG
            assert(x_coord + i < n);
#endif
            float *curr_work_pos = matrix + (ptrdiff_t)(x_coord + i) * lda + (ptrdiff_t)row_offset;
            for (int j = 0; j < unvec_rows; j++) {
#ifdef EINSUMS_DEBUG
                assert(row_offset + j < m);
#endif
                curr_work_pos[j] *= mult;
            }
        }
    }
}

__global__ void dlascl_full_matrix(double mult, int m, int n, double *matrix, int lda) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    // Vectorization for the win!
    double4 mult_vec = make_double4(mult, mult, mult, mult);

    // Find out where the thread is in the block.
    int thread_x_coord = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_rows    = blockDim.y * gridDim.y;
    int thread_cols    = blockDim.x * gridDim.x;

    // Figure out how many elements each thread can handle.
    int work_rows = m / (4 * thread_rows);
    int work_cols = n / thread_cols;

    // If there are some remaining, figure out how many.
    int remaining_rows = m % (4 * thread_rows);
    int remaining_cols = n % thread_cols;

    // Find the number of extra vectorable columns and unvectorable columns.
    int extra_vec_rows = remaining_rows / 4;
    int unvec_rows     = remaining_rows % 4;

    int x_coord = thread_x_coord * work_rows;
    int y_coord = thread_y_coord * work_cols;

    // If the thread's row is less than the number of remaining rows, then assign it the work of one of the remaining rows.
    if (thread_y_coord < extra_vec_rows) {
        work_rows++;
        y_coord++;
    } else {
        y_coord += extra_vec_rows;
    }

    if (thread_x_coord < remaining_cols) {
        work_cols++;
        x_coord++;
    } else {
        x_coord += remaining_cols;
    }

    // If the thread does not have any work, exit.
    if (work_rows == 0 || work_cols == 0) {
        return;
    }

    if (x_coord >= n || y_coord >= m) {
        return;
    }

    // If the thread's

    // Now, scale.

    // Start with the vectorable rows and columns.
    for (int i = 0; i < work_cols; i++) {
        double4 *curr_work_pos = (double4 *)matrix + (ptrdiff_t)(x_coord + i) * lda + (ptrdiff_t)y_coord;
#ifdef EINSUMS_DEBUG
        assert(x_coord + i < n);
#endif

        for (int j = 0; j < work_rows; j++) {
#ifdef EINSUMS_DEBUG
            assert(y_coord + j < m);
#endif
            curr_work_pos[j] *= mult_vec;
        }
    }

    // Then do the unvectorable columns.
    if (threadIdx.x == 0) {
        int row_offset = m - (m % 4);
        for (int i = 0; i < work_cols; i++) {
#ifdef EINSUMS_DEBUG
            assert(x_coord + i < n);
#endif
            double *curr_work_pos = matrix + (ptrdiff_t)(x_coord + i) * lda + (ptrdiff_t)row_offset;
            for (int j = 0; j < unvec_rows; j++) {
#ifdef EINSUMS_DEBUG
                assert(row_offset + j < m);
#endif
                curr_work_pos[j] *= mult;
            }
        }
    }
}

int slascl(char type, int kl, int ku, float cfrom, float cto, int m, int n, float *vec, int lda) {
    LabeledSection0();

    int info = 0;

    int compute_type = -1;

    // Check to see if the type is accepted.
    switch (type) {
    case 'g':
    case 'G':
        compute_type = 0;
        break;
    case 'l': // ell not one.
    case 'L':
        compute_type = 1;
        break;
    case 'u':
    case 'U':
        compute_type = 2;
        break;
    case 'h':
    case 'H':
        compute_type = 3;
        break;
    case 'b':
    case 'B':
        compute_type = 4;
        break;
    case 'q':
    case 'Q':
        compute_type = 5;
        break;
    case 'z':
    case 'Z':
        compute_type = 6;
        break;
    default:
        return -1;
    }

    // Check the denominator. Can't be zero or NaN.
    if (cfrom == 0.0f || std::isnan(cfrom)) {
        return -4;
    }

    // Check the numerator. Can't be NaN.
    if (std::isnan(cto)) {
        return -5;
    }

    // Check the number of rows. Can't be negative.
    if (m < 0) {
        return -6;
    }

    // Check the number of columns. Can't be negative. Sometimes has more restrictions.
    if (n < 0 || (compute_type == 4 && n != m) || (compute_type == 5 && n != m)) {
        return -7;
    }

    // Check the leading dimension.
    if (compute_type == 3 && lda < std::max(1, m)) {
        return -9;
    }

    if (compute_type >= 4) {
        if (kl < 0 || kl > std::max(m - 1, 0)) {
            return -2;
        }

        if (ku < 0 || ku > std::max(n - 1, 0) || ((compute_type == 4 || compute_type == 5) && kl != ku)) {
            return -3;
        }

        if ((compute_type == 4 && lda < kl + 1) || (compute_type == 5 && lda < ku + 1) || (compute_type == 6 && lda < 2 * kl + ku + 1)) {
            return -9;
        }
    }

    // If the size of the compute is zero, exit.
    if (n == 0 || m == 0) {
        return 0;
    }

    constexpr float sfmin  = std::numeric_limits<float>::min();
    constexpr float small  = 1.0f / std::numeric_limits<float>::max();
    constexpr float smlnum = (small > sfmin) ? small * (1.0f + std::numeric_limits<float>::epsilon()) : sfmin;
    constexpr float bignum = 1.0f / smlnum;

    float from_temp = cfrom;
    float to_temp   = cto;
    bool  finished  = false;

    do {
        float cfrom1 = from_temp * smlnum;
        float mult;

        // Check for infinity.
        if (cfrom1 == from_temp) {
            mult     = to_temp / from_temp;
            finished = true;
        } else {
            float cto1 = to_temp / bignum;

            // Check for infinity.
            if (cto1 == to_temp) {
                mult      = to_temp;
                finished  = true;
                from_temp = 1.0f;
            } else if (std::abs(cfrom1) > std::abs(to_temp) && to_temp != 0.0f) {
                mult      = smlnum;
                finished  = false;
                from_temp = cfrom1;
            } else if (std::abs(cto1) > std::abs(from_temp)) {
                mult     = bignum;
                finished = false;
                to_temp  = cto1;
            } else {
                mult     = to_temp / from_temp;
                finished = true;
                // If we multiply by 1, then don't actually do the multiplication.
                if (mult == 1.0f) {
                    return 0;
                }
            }
        }

        float *mult_gpu;
        int    loop_bound;
        int    k_one, k2, k3, k4;

        switch (compute_type) {
        case 0:
            if (lda == m) {
                sscal(m * n, mult, vec, 1);
            } else {
                dim3 row_block_dims = gpu::block_size(m);
                dim3 row_blocks     = gpu::blocks(m);
                dim3 col_block_dims = gpu::block_size(n);
                dim3 col_blocks     = gpu::blocks(n);

                dim3 block_dims{col_block_dims.x, row_block_dims.x, 1};
                dim3 blocks{col_blocks.x, row_blocks.x, 1};

                slascl_full_matrix<<<block_dims, blocks, 0, gpu::get_stream()>>>(mult, m, n, vec, lda);
                gpu::stream_wait();
            }
            break;
        case 1:
            mult_gpu   = gpu::register_host_variable(mult);
            loop_bound = std::min(n, m);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < loop_bound; j++) {
                hipblas_catch(hipblasSscal(gpu::get_blas_handle(), m - j, mult_gpu, vec + (ptrdiff_t)j * lda + j, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 2:
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                hipblas_catch(hipblasSscal(gpu::get_blas_handle(), std::min(j, m), mult_gpu, vec + (ptrdiff_t)j * lda, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 3:
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                hipblas_catch(hipblasSscal(gpu::get_blas_handle(), std::min(j + 1, m), mult_gpu, vec + (ptrdiff_t)j * lda, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 4:
            k3 = kl;
            k4 = n;

            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                hipblas_catch(
                    hipblasSscal(gpu::get_blas_handle(), std::max(std::min(k3, k4 - j), 0), mult_gpu, vec + (ptrdiff_t)j * lda, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 5:
            k_one    = ku + 1;
            k3       = ku;
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                int lower_bound = std::max(k_one - j, 0);
                hipblas_catch(hipblasSscal(gpu::get_blas_handle(), std::max(k3 - lower_bound, 0), mult_gpu,
                                           vec + (ptrdiff_t)j * lda + lower_bound, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 6:
            k_one    = kl + ku + 1;
            k2       = kl;
            k3       = 2 * kl + ku;
            k4       = kl + ku + m;
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                int lower_bound = std::max(k_one - j, k2);
                int upper_bound = std::max(k3, k4 - j);
                hipblas_catch(hipblasSscal(gpu::get_blas_handle(), std::max(upper_bound - lower_bound, 0), mult_gpu,
                                           vec + (ptrdiff_t)j * lda + lower_bound, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        }
    } while (!finished);

    return 0;
}

int dlascl(char type, int kl, int ku, double cfrom, double cto, int m, int n, double *vec, int lda) {
    LabeledSection0();

    LabeledSection0();

    int info = 0;

    int compute_type = -1;

    // Check to see if the type is accepted.
    switch (type) {
    case 'g':
    case 'G':
        compute_type = 0;
        break;
    case 'l': // ell not one.
    case 'L':
        compute_type = 1;
        break;
    case 'u':
    case 'U':
        compute_type = 2;
        break;
    case 'h':
    case 'H':
        compute_type = 3;
        break;
    case 'b':
    case 'B':
        compute_type = 4;
        break;
    case 'q':
    case 'Q':
        compute_type = 5;
        break;
    case 'z':
    case 'Z':
        compute_type = 6;
        break;
    default:
        return -1;
    }

    // Check the denominator. Can't be zero or NaN.
    if (cfrom == 0.0f || std::isnan(cfrom)) {
        return -4;
    }

    // Check the numerator. Can't be NaN.
    if (std::isnan(cto)) {
        return -5;
    }

    // Check the number of rows. Can't be negative.
    if (m < 0) {
        return -6;
    }

    // Check the number of columns. Can't be negative. Sometimes has more restrictions.
    if (n < 0 || (compute_type == 4 && n != m) || (compute_type == 5 && n != m)) {
        return -7;
    }

    // Check the leading dimension.
    if (compute_type == 3 && lda < std::max(1, m)) {
        return -9;
    }

    if (compute_type >= 4) {
        if (kl < 0 || kl > std::max(m - 1, 0)) {
            return -2;
        }

        if (ku < 0 || ku > std::max(n - 1, 0) || ((compute_type == 4 || compute_type == 5) && kl != ku)) {
            return -3;
        }

        if ((compute_type == 4 && lda < kl + 1) || (compute_type == 5 && lda < ku + 1) || (compute_type == 6 && lda < 2 * kl + ku + 1)) {
            return -9;
        }
    }

    // If the size of the compute is zero, exit.
    if (n == 0 || m == 0) {
        return 0;
    }

    constexpr double sfmin  = std::numeric_limits<double>::min();
    constexpr double small  = 1.0f / std::numeric_limits<double>::max();
    constexpr double smlnum = (small > sfmin) ? small * (1.0f + std::numeric_limits<double>::epsilon()) : sfmin;
    constexpr double bignum = 1.0f / smlnum;

    double from_temp = cfrom;
    double to_temp   = cto;
    bool   finished  = false;

    do {
        double cfrom1 = from_temp * smlnum;
        double mult;

        // Check for infinity.
        if (cfrom1 == from_temp) {
            mult     = to_temp / from_temp;
            finished = true;
        } else {
            double cto1 = to_temp / bignum;

            // Check for infinity.
            if (cto1 == to_temp) {
                mult      = to_temp;
                finished  = true;
                from_temp = 1.0f;
            } else if (std::abs(cfrom1) > std::abs(to_temp) && to_temp != 0.0f) {
                mult      = smlnum;
                finished  = false;
                from_temp = cfrom1;
            } else if (std::abs(cto1) > std::abs(from_temp)) {
                mult     = bignum;
                finished = false;
                to_temp  = cto1;
            } else {
                mult     = to_temp / from_temp;
                finished = true;
                // If we multiply by 1, then don't actually do the multiplication.
                if (mult == 1.0f) {
                    return 0;
                }
            }
        }

        double *mult_gpu;
        int     loop_bound;
        int     k_one, k2, k3, k4;

        switch (compute_type) {
        case 0:
            if (lda == m) {
                dscal(m * n, mult, vec, 1);
            } else {
                dim3 row_block_dims = gpu::block_size(m);
                dim3 row_blocks     = gpu::blocks(m);
                dim3 col_block_dims = gpu::block_size(n);
                dim3 col_blocks     = gpu::blocks(n);

                dim3 block_dims{col_block_dims.x, row_block_dims.x, 1};
                dim3 blocks{col_blocks.x, row_blocks.x, 1};

                dlascl_full_matrix<<<block_dims, blocks, 0, gpu::get_stream()>>>(mult, m, n, vec, lda);
                gpu::stream_wait();
            }
            break;
        case 1:
            mult_gpu   = gpu::register_host_variable(mult);
            loop_bound = std::min(n, m);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < loop_bound; j++) {
                hipblas_catch(hipblasDscal(gpu::get_blas_handle(), m - j, mult_gpu, vec + (ptrdiff_t)j * lda + j, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 2:
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                hipblas_catch(hipblasDscal(gpu::get_blas_handle(), std::min(j, m), mult_gpu, vec + (ptrdiff_t)j * lda, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 3:
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                hipblas_catch(hipblasDscal(gpu::get_blas_handle(), std::min(j + 1, m), mult_gpu, vec + (ptrdiff_t)j * lda, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 4:
            k3 = kl;
            k4 = n;

            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                hipblas_catch(
                    hipblasDscal(gpu::get_blas_handle(), std::max(std::min(k3, k4 - j), 0), mult_gpu, vec + (ptrdiff_t)j * lda, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 5:
            k_one    = ku + 1;
            k3       = ku;
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                int lower_bound = std::max(k_one - j, 0);
                hipblas_catch(hipblasDscal(gpu::get_blas_handle(), std::max(k3 - lower_bound, 0), mult_gpu,
                                           vec + (ptrdiff_t)j * lda + lower_bound, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        case 6:
            k_one    = kl + ku + 1;
            k2       = kl;
            k3       = 2 * kl + ku;
            k4       = kl + ku + m;
            mult_gpu = gpu::register_host_variable(mult);
            EINSUMS_OMP_PARALLEL_FOR
            for (int j = 0; j < n; j++) {
                int lower_bound = std::max(k_one - j, k2);
                int upper_bound = std::max(k3, k4 - j);
                hipblas_catch(hipblasDscal(gpu::get_blas_handle(), std::max(upper_bound - lower_bound, 0), mult_gpu,
                                           vec + (ptrdiff_t)j * lda + lower_bound, 1));
            }
            gpu::all_stream_wait();
            gpu::unregister_host_variable(mult_gpu);
            break;
        }
    } while (!finished);

    return 0;
}

} // namespace einsums::blas::hip