//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <list>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <omp.h>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUStreams/GPUStreams.hpp>
#    include <Einsums/TensorAlgebra/Backends/GPUTensorAlgebra.hpp>
#    include <Einsums/TypeSupport/GPUCast.hpp>

#    include <EinsumsPy/GPU/PyGPUView.hpp>
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#    include <hipblas/hipblas.h>
#endif

#include <cstddef>
#include <string>
#include <vector>

namespace einsums::tensor_algebra {

namespace detail {
EINSUMS_EXPORT std::vector<size_t> get_dim_ranges_for_many(pybind11::buffer_info const &C, std::vector<int> const &C_perm,
                                                           pybind11::buffer_info const &A, std::vector<int> const &A_perm,
                                                           pybind11::buffer_info const &B, std::vector<int> const &B_perm,
                                                           int unique_indices);
EINSUMS_EXPORT std::vector<size_t> get_dim_ranges_for_many(pybind11::buffer_info const &A, std::vector<int> const &A_perm,
                                                           pybind11::buffer_info const &B, std::vector<int> const &B_perm,
                                                           int unique_indices);

EINSUMS_EXPORT std::string intersect(std::string const &st1, std::string const &st2);

template <typename T>
std::list<T> intersect(std::vector<T> const &vec1, std::vector<T> const &vec2) {
    std::list<T> out;

    for (int i = 0; i < vec1.size(); i++) {
        for (int j = 0; j < vec2.size(); j++) {
            if (vec1[i] == vec2[j]) {
                out.push_back(vec1[i]);
            }
        }
    }
    return out;
}

#ifdef EINSUMS_COMPUTE_CODE

EINSUMS_EXPORT std::vector<size_t> get_dim_ranges_for_many(python::PyGPUView const &C, std::vector<int> const &C_perm,
                                                           python::PyGPUView const &A, std::vector<int> const &A_perm,
                                                           python::PyGPUView const &B, std::vector<int> const &B_perm, int unique_indices);

EINSUMS_EXPORT std::vector<size_t> get_dim_ranges_for_many(python::PyGPUView const &A, std::vector<int> const &A_perm,
                                                           python::PyGPUView const &B, std::vector<int> const &B_perm, int unique_indices);
/**
 * Perform the generic algorithm on the GPU.
 */
template <typename DataType>
__global__ void einsum_generic_algorithm_gpu(size_t const *__restrict__ unique_strides, size_t const *__restrict__ C_index_strides,
                                             int const *__restrict__ C_index_table, int const *__restrict__ A_index_table,
                                             int const *__restrict__ B_index_table, DataType const C_prefactor, DataType *__restrict__ C,
                                             size_t const *__restrict__ C_stride, size_t const *__restrict__ C_stride_unique,
                                             DataType const AB_prefactor, DataType const *__restrict__ A,
                                             size_t const *__restrict__ A_stride, size_t const *__restrict__ A_stride_unique,
                                             DataType const *__restrict__ B, size_t const *__restrict__ B_stride,
                                             size_t const *__restrict__ B_stride_unique, size_t max_index, size_t C_size, size_t C_rank,
                                             size_t A_rank, size_t B_rank, size_t unique_rank) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_sentinel, B_sentinel, C_sentinel;

    // First, set C.
    if (is_zero(C_prefactor)) {
        for (size_t index = thread_id; index < C_size; index += kernel_size) {
            size_t C_index = 0, quotient = index;
            for (int i = 0; i < C_rank; i++) {
                C_index += (C_stride[i] / sizeof(DataType)) * (quotient / C_index_strides[i]);
                quotient %= C_index_strides[i];
            }
            make_zero(C[C_index]);
        }
    } else {
        for (size_t index = thread_id; index < C_size; index += kernel_size) {
            size_t C_index = 0, quotient = index;
            for (int i = 0; i < C_rank; i++) {
                C_index += (C_stride[i] / sizeof(DataType)) * (quotient / C_index_strides[i]);
                quotient %= C_index_strides[i];
            }
            C[C_index] = C[C_index] * C_prefactor;
        }
    }

    __syncthreads();

    // Now, contract.
    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
        size_t quotient = curr_index;
        A_sentinel      = 0;
        B_sentinel      = 0;
        C_sentinel      = 0;
        for (int i = 0; i < unique_rank; i++) {
            size_t unique_index = quotient / unique_strides[i];
            quotient %= unique_strides[i];

            A_sentinel += (A_stride_unique[i] / sizeof(DataType)) * unique_index;
            B_sentinel += (B_stride_unique[i] / sizeof(DataType)) * unique_index;
            C_sentinel += (C_stride_unique[i] / sizeof(DataType)) * unique_index;
        }

        einsums::gpu::atomicAdd_wrap(C + C_sentinel, gpu_ops::mult(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel]));
    }
}

/**
 * Perform the generic algorithm on the GPU. This is the special case where each element in C will be seen exactly once, so we can
 * skip synchronizations.
 */
template <typename DataType>
__global__ void einsum_generic_algorithm_direct_product_gpu(
    size_t const *__restrict__ unique_strides, size_t const *__restrict__ C_index_strides, int const *__restrict__ C_index_table,
    int const *__restrict__ A_index_table, int const *__restrict__ B_index_table, DataType const    C_prefactor, DataType *__restrict__ C,
    size_t const *__restrict__ C_stride, size_t const *__restrict__ C_stride_unique, DataType const AB_prefactor,
    DataType const *__restrict__ A, size_t const *__restrict__ A_stride, size_t const *__restrict__ A_stride_unique,
    DataType const *__restrict__ B, size_t const *__restrict__ B_stride, size_t const *__restrict__ B_stride_unique, size_t max_index,
    size_t C_size, size_t C_rank, size_t A_rank, size_t B_rank, size_t unique_rank) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_sentinel, B_sentinel, C_sentinel;

    // Now, contract.
    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
        size_t quotient = curr_index;
        A_sentinel      = 0;
        B_sentinel      = 0;
        C_sentinel      = 0;
        for (int i = 0; i < unique_rank; i++) {
            size_t unique_index = quotient / unique_strides[i];
            quotient %= unique_strides[i];

            A_sentinel += (A_stride_unique[i] / sizeof(DataType)) * unique_index;
            B_sentinel += (B_stride_unique[i] / sizeof(DataType)) * unique_index;
            C_sentinel += (C_stride_unique[i] / sizeof(DataType)) * unique_index;
        }

        // We can do this here since we are guaranteed to see each element only once.
        if (is_zero(C_prefactor)) {
            C[C_sentinel] = gpu_ops::mult(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel]);
        } else {
            C[C_sentinel] =
                gpu_ops::fma(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel], gpu_ops::mult(C_prefactor, C[C_sentinel]));
        }
    }
}

/**
 * Compute kernel that runs when C has a rank of zero. There are some optimizations that can be made in this case.
 */
template <typename DataType>
__global__ void einsum_generic_zero_rank_gpu(size_t const *__restrict__ unique_strides, int const *__restrict__ A_index_table,
                                             int const *__restrict__ B_index_table, DataType *__restrict__ C, DataType const AB_prefactor,
                                             DataType const *__restrict__ A, size_t const *__restrict__ A_stride,
                                             size_t const *__restrict__ A_stride_unique, DataType const *__restrict__ B,
                                             size_t const *__restrict__ B_stride, size_t const *__restrict__ B_stride_unique,
                                             size_t max_index, size_t A_rank, size_t B_rank, size_t unique_rank) {

    DataType value;

    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    // Clear the dot product.
    make_zero(value);

    if (thread_id == 0) {
        make_zero(*C);
    }

    __syncthreads();

    size_t A_sentinel, B_sentinel;

    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
        size_t quotient = curr_index;
        A_sentinel      = 0;
        B_sentinel      = 0;
        for (int i = 0; i < unique_rank; i++) {
            size_t unique_index = quotient / unique_strides[i];
            quotient %= unique_strides[i];

            A_sentinel += (A_stride_unique[i] / sizeof(DataType)) * unique_index;
            B_sentinel += (B_stride_unique[i] / sizeof(DataType)) * unique_index;
        }

        value = gpu_ops::fma(A[A_sentinel], B[B_sentinel], value);
    }

    einsums::gpu::atomicAdd_wrap(C, gpu_ops::mult(AB_prefactor, value));
}

/**
 * Compute the dot product on GPU.
 */
template <typename T>
__global__ void dot_kernel(size_t const *unique_strides, T *__restrict__ C, T const *__restrict__ A, size_t const *__restrict__ A_strides,
                           T const *__restrict__ B, size_t const *__restrict__ B_strides, size_t max_index, size_t rank) {
    T value;

    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    // Clear the dot product.
    make_zero(value);

    if (thread_id == 0) {
        make_zero(*C);
    }

    __syncthreads();

    size_t A_sentinel, B_sentinel;

    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
        size_t quotient = curr_index;
        A_sentinel      = 0;
        B_sentinel      = 0;
        for (int i = 0; i < rank; i++) {
            size_t unique_index = quotient / unique_strides[i];
            quotient %= unique_strides[i];

            A_sentinel += (A_strides[i] / sizeof(T)) * unique_index;
            B_sentinel += (B_strides[i] / sizeof(T)) * unique_index;
        }

        value = gpu_ops::fma(A[A_sentinel], B[B_sentinel], value);
    }

    einsums::gpu::atomicAdd_wrap(C, value);
}

/**
 * Compute the direct product on GPU.
 */
template <typename DataType>
__global__ void direct_product_kernel(size_t const *__restrict__ index_strides, DataType const C_prefactor, DataType *__restrict__ C,
                                      size_t const *__restrict__ C_stride, DataType const      AB_prefactor, DataType const *__restrict__ A,
                                      size_t const *__restrict__ A_stride, DataType const *__restrict__ B,
                                      size_t const *__restrict__ B_stride, size_t max_index, size_t rank) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t A_sentinel, B_sentinel, C_sentinel;

    // Now, contract.
    for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
        size_t quotient = curr_index;
        A_sentinel      = 0;
        B_sentinel      = 0;
        C_sentinel      = 0;
        for (int i = 0; i < rank; i++) {
            size_t unique_index = quotient / index_strides[i];
            quotient %= index_strides[i];

            A_sentinel += (A_stride[i] / sizeof(DataType)) * unique_index;
            B_sentinel += (B_stride[i] / sizeof(DataType)) * unique_index;
            C_sentinel += (C_stride[i] / sizeof(DataType)) * unique_index;
        }

        // We can do this here since we are guaranteed to see each element only once.
        if (is_zero(C_prefactor)) {
            C[C_sentinel] = gpu_ops::mult(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel]);
        } else {
            C[C_sentinel] =
                gpu_ops::fma(gpu_ops::mult(AB_prefactor, A[A_sentinel]), B[B_sentinel], gpu_ops::mult(C_prefactor, C[C_sentinel]));
        }
    }
}

/**
 * Scale a tensor on GPU.
 */
template <typename T>
__global__ void scale_array(T factor, T *__restrict__ array, size_t max_index) {
    using namespace einsums::gpu;

    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    if (is_zero(factor)) {
        for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
            make_zero(array[curr_index]);
        }
    } else {

        for (size_t curr_index = thread_id; curr_index < max_index; curr_index += kernel_size) {
            array[curr_index] = gpu_ops::mult(factor, array[curr_index]);
        }
    }
}
#endif

} // namespace detail

/**
 * @class PyEinsumGenericPlan
 *
 * @brief Holds the info for the generic algorithm, called when all the optimizations fail.
 */
class EINSUMS_EXPORT PyEinsumGenericPlan {
  protected:
    std::vector<int> _C_permute, _A_permute, _B_permute;
    int              _num_inds;
    bool             _direct_product_swap;

#ifdef EINSUMS_COMPUTE_CODE
    /**
     * Set up and execute the GPU code.
     */
    template <typename T>
    void execute_generic_gpu(T C_prefactor, python::PyGPUView &C, T AB_prefactor, python::PyGPUView const &A,
                             python::PyGPUView const &B) const {
        std::string T_spec = pybind11::format_descriptor<T>::format();
        if (T_spec != C.fmt_spec() || T_spec != A.fmt_spec() || T_spec != B.fmt_spec()) {
            EINSUMS_THROW_EXCEPTION(pybind11::type_error, "Can not mix data types on the GPU!");
        }

        using namespace einsums::tensor_algebra::detail;
        if (_C_permute.size() != C.rank() || _A_permute.size() != A.rank() || _B_permute.size() != B.rank()) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Tensor ranks do not match the indices!");
        }

        std::vector<size_t> unique_dims = get_dim_ranges_for_many(A, _A_permute, B, _B_permute, _num_inds);

        std::vector<size_t> unique_strides, C_index_strides, A_unique_stride, B_unique_stride, C_unique_stride;

        dims_to_strides(unique_dims, unique_strides);
        dims_to_strides(C.dims(), C_index_strides);

        A_unique_stride.resize(unique_strides.size());
        B_unique_stride.resize(unique_strides.size());
        C_unique_stride.resize(unique_strides.size());

        for (int i = 0; i < A_unique_stride.size(); i++) {
            bool found = false;
            for (int j = 0; j < _A_permute.size(); j++) {
                if (_A_permute[j] == i) {
                    A_unique_stride[i] = A.stride(j);
                    found              = true;
                    break;
                }
            }
            if (!found) {
                A_unique_stride[i] = 0;
            }
        }

        for (int i = 0; i < B_unique_stride.size(); i++) {
            bool found = false;
            for (int j = 0; j < _B_permute.size(); j++) {
                if (_B_permute[j] == i) {
                    B_unique_stride[i] = B.stride(j);
                    found              = true;
                    break;
                }
            }
            if (!found) {
                B_unique_stride[i] = 0;
            }
        }

        for (int i = 0; i < C_unique_stride.size(); i++) {
            bool found = false;
            for (int j = 0; j < _C_permute.size(); j++) {
                if (_C_permute[j] == i) {
                    C_unique_stride[i] = C.stride(j);
                    found              = true;
                    break;
                }
            }
            if (!found) {
                C_unique_stride[i] = 0;
            }
        }

        int    *gpu_C_index_table, *gpu_A_index_table, *gpu_B_index_table;
        size_t *gpu_unique_strides, *gpu_C_index_strides;

        size_t *gpu_C_dims, *gpu_C_stride, *gpu_A_stride, *gpu_B_stride, *gpu_A_unique_stride, *gpu_B_unique_stride, *gpu_C_unique_stride;

        if (_C_permute.size() != 0) {
            hip_catch(hipMalloc((void **)&gpu_C_unique_stride, unique_strides.size() * sizeof(size_t)));
        }
        hip_catch(hipMalloc((void **)&gpu_A_unique_stride, unique_strides.size() * sizeof(size_t)));
        hip_catch(hipMalloc((void **)&gpu_B_unique_stride, unique_strides.size() * sizeof(size_t)));

        hip_catch(hipMalloc((void **)&gpu_unique_strides, _num_inds * sizeof(size_t)));
        if (_C_permute.size() != 0) {
            hip_catch(hipMalloc((void **)&gpu_C_index_strides, _C_permute.size() * sizeof(size_t)));

            hip_catch(hipMalloc((void **)&gpu_C_index_table, _C_permute.size() * sizeof(int)));
        }
        hip_catch(hipMalloc((void **)&gpu_A_index_table, _A_permute.size() * sizeof(int)));
        hip_catch(hipMalloc((void **)&gpu_B_index_table, _B_permute.size() * sizeof(int)));

        hip_catch(
            hipMemcpy((void *)gpu_A_index_table, (void const *)_A_permute.data(), _A_permute.size() * sizeof(int), hipMemcpyHostToDevice));
        hip_catch(
            hipMemcpy((void *)gpu_B_index_table, (void const *)_B_permute.data(), _B_permute.size() * sizeof(int), hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)gpu_A_unique_stride, (void const *)A_unique_stride.data(), A_unique_stride.size() * sizeof(size_t),
                            hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)gpu_B_unique_stride, (void const *)B_unique_stride.data(), B_unique_stride.size() * sizeof(size_t),
                            hipMemcpyHostToDevice));

        if (_C_permute.size() != 0) {
            hip_catch(hipMemcpy((void *)gpu_C_index_table, (void const *)_C_permute.data(), _C_permute.size() * sizeof(int),
                                hipMemcpyHostToDevice));
            hip_catch(hipMemcpy((void *)gpu_C_unique_stride, (void const *)C_unique_stride.data(), C_unique_stride.size() * sizeof(size_t),
                                hipMemcpyHostToDevice));
        }

        hip_catch(hipMemcpy((void *)gpu_unique_strides, (void const *)unique_strides.data(), unique_strides.size() * sizeof(size_t),
                            hipMemcpyHostToDevice));

        if (_C_permute.size() != 0) {
            hip_catch(hipMemcpy((void *)gpu_C_index_strides, (void const *)C_index_strides.data(), _C_permute.size() * sizeof(size_t),
                                hipMemcpyHostToDevice));
        }

        auto threads = gpu::block_size(unique_dims[0] * unique_strides[0]), blocks = gpu::blocks(unique_dims[0] * unique_strides[0]);

        if (_C_permute.size() != 0) {
            if (!_direct_product_swap) {
                detail::einsum_generic_algorithm_gpu<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                    gpu_unique_strides, gpu_C_index_strides, gpu_C_index_table, gpu_A_index_table, gpu_B_index_table,
                    HipCast<DevDatatype<T>, T>::cast(C_prefactor), (DevDatatype<T> *)C.dev_data(), C.gpu_strides(), gpu_C_unique_stride,
                    HipCast<DevDatatype<T>, T>::cast(AB_prefactor), (DevDatatype<T> *)A.dev_data(), A.gpu_strides(), gpu_A_unique_stride,
                    (DevDatatype<T> *)B.dev_data(), B.gpu_strides(), gpu_B_unique_stride, unique_dims[0] * unique_strides[0], C.size(),
                    C.rank(), A.rank(), B.rank(), unique_dims.size());
            } else {
                detail::einsum_generic_algorithm_direct_product_gpu<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                    gpu_unique_strides, gpu_C_index_strides, gpu_C_index_table, gpu_A_index_table, gpu_B_index_table,
                    HipCast<DevDatatype<T>, T>::cast(C_prefactor), (DevDatatype<T> *)C.dev_data(), C.gpu_strides(), gpu_C_unique_stride,
                    HipCast<DevDatatype<T>, T>::cast(AB_prefactor), (DevDatatype<T> *)A.dev_data(), A.gpu_strides(), gpu_A_unique_stride,
                    (DevDatatype<T> *)B.dev_data(), B.gpu_strides(), gpu_B_unique_stride, unique_dims[0] * unique_strides[0], C.size(),
                    C.rank(), A.rank(), B.rank(), unique_dims.size());
            }

            gpu::stream_wait();

            hip_catch(hipFree((void *)gpu_C_unique_stride));
            hip_catch(hipFree((void *)gpu_C_index_table));
            hip_catch(hipFree((void *)gpu_C_index_strides));
        } else {
            C.update_D2H(); // This expects that the most up-to-date data is on the device, but we need to work in core for this part.
            T &C_val = *(T *)C.host_data();

            if (C_prefactor == T{0.0}) {
                C_val = T{0.0};
            } else {
                C_val *= C_prefactor;
            }

            C.update_H2D();

            detail::einsum_generic_zero_rank_gpu<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                gpu_unique_strides, gpu_A_index_table, gpu_B_index_table, (DevDatatype<T> *)C.dev_data(),
                HipCast<DevDatatype<T>, T>::cast(AB_prefactor), (DevDatatype<T> *)A.dev_data(), A.gpu_strides(), gpu_A_unique_stride,
                (DevDatatype<T> *)B.dev_data(), B.gpu_strides(), gpu_B_unique_stride, unique_dims[0] * unique_strides[0], _A_permute.size(),
                _B_permute.size(), _num_inds);

            gpu::stream_wait();
        }

        hip_catch(hipFree((void *)gpu_A_index_table));
        hip_catch(hipFree((void *)gpu_B_index_table));
        hip_catch(hipFree((void *)gpu_unique_strides));
        hip_catch(hipFree((void *)gpu_A_unique_stride));
        hip_catch(hipFree((void *)gpu_B_unique_stride));
    }
#endif

    /**
     * Execute the generic algorithm.
     */
    template <typename T>
    void execute_generic(T C_prefactor, pybind11::buffer &C, T AB_prefactor, pybind11::buffer const &A, pybind11::buffer const &B) const {
        using namespace einsums::tensor_algebra::detail;

        pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);

        if (_C_permute.size() != C_info.ndim || _A_permute.size() != A_info.ndim || _B_permute.size() != B_info.ndim) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Tensor ranks do not match the indices!");
        }
        std::vector<size_t> unique_dims =
            detail::get_dim_ranges_for_many(C_info, _C_permute, A_info, _A_permute, B_info, _B_permute, _num_inds);
        std::vector<size_t> unique_strides, C_index_strides, C_dims(_C_permute.size());

        dims_to_strides(unique_dims, unique_strides);

        T       *C_data = (T *)C_info.ptr;
        T const *A_data = (T const *)A_info.ptr;
        T const *B_data = (T const *)B_info.ptr;

        for (int i = 0; i < _C_permute.size(); i++) {
            C_dims[i] = C_info.shape[i];
        }

        dims_to_strides(C_dims, C_index_strides);

        if (C_prefactor == T{0.0}) {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t sentinel = 0; sentinel < C_dims[0] * C_index_strides[0]; sentinel++) {
                size_t quotient = sentinel;
                size_t C_index  = 0;
                for (int i = 0; i < _C_permute.size(); i++) {
                    C_index += (C_info.strides[i] / sizeof(T)) * (quotient / C_index_strides[i]);
                    quotient %= C_index_strides[i];
                }

                C_data[C_index] = T{0.0};
            }
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t sentinel = 0; sentinel < C_dims[0] * C_index_strides[0]; sentinel++) {
                size_t quotient = sentinel;
                size_t C_index  = 0;
                for (int i = 0; i < _C_permute.size(); i++) {
                    C_index += (C_info.strides[i] / sizeof(T)) * (quotient / C_index_strides[i]);
                    quotient %= C_index_strides[i];
                }

                C_data[C_index] *= C_prefactor;
            }
        }

        EINSUMS_OMP_PARALLEL {
            auto thread = omp_get_thread_num();
            auto kernel = omp_get_num_threads();

            std::vector<size_t> A_index(_A_permute.size()), B_index(_B_permute.size()), C_index(_C_permute.size()),
                unique_index(unique_dims.size());

            for (size_t sentinel = thread; sentinel < unique_dims[0] * unique_strides[0]; sentinel += kernel) {
                size_t quotient = sentinel;

                for (int i = 0; i < unique_index.size(); i++) {
                    unique_index[i] = quotient / unique_strides[i];
                    quotient %= unique_strides[i];
                }

                size_t A_ord = 0, B_ord = 0, C_ord = 0;

                for (int i = 0; i < _A_permute.size(); i++) {
                    A_ord += (A_info.strides[i] / sizeof(T)) * unique_index[_A_permute[i]];
                }

                for (int i = 0; i < _B_permute.size(); i++) {
                    B_ord += (B_info.strides[i] / sizeof(T)) * unique_index[_B_permute[i]];
                }

                for (int i = 0; i < _C_permute.size(); i++) {
                    C_ord += (C_info.strides[i] / sizeof(T)) * unique_index[_C_permute[i]];
                }

                C_data[C_ord] += AB_prefactor * A_data[A_ord] * B_data[B_ord];
            }
        }
    }

  public:
    PyEinsumGenericPlan() = delete;
    PyEinsumGenericPlan(int num_inds, std::vector<int> C_permute, std::vector<int> A_permute, std::vector<int> B_permute);
    PyEinsumGenericPlan(PyEinsumGenericPlan const &) = default;
    ~PyEinsumGenericPlan()                           = default;

#ifdef EINSUMS_COMPUTE_CODE
    /**
     * Execute the specified plan on the given tensors. This one uses the GPU algorithm.
     */
    virtual void execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                         python::PyGPUView const &A, python::PyGPUView const &B) const;
#endif

    /**
     * Execute the specified plan on the given tensors.
     */
    virtual void execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                         pybind11::buffer const &A, pybind11::buffer const &B) const;
};

class EINSUMS_EXPORT PyEinsumDotPlan : public PyEinsumGenericPlan {
  private:
#ifdef EINSUMS_COMPUTE_CODE
    template <typename T>
    void execute_imp_gpu(T C_prefactor, python::PyGPUView &C, T AB_prefactor, python::PyGPUView const &A,
                         python::PyGPUView const &B) const {
        if (A.rank() != B.rank()) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Tensor ranks do not match the indices!");
        }

        std::vector<size_t> unique_strides(A.rank());
        size_t              prod = 1;

        for (int i = A.rank() - 1; i >= 0; i--) {
            unique_strides[i] = prod;
            prod *= A.dim(i);
        }

        size_t *gpu_unique_strides;

        hip_catch(hipMalloc((void **)&gpu_unique_strides, unique_strides.size() * sizeof(size_t)));
        hip_catch(hipMemcpy((void *)gpu_unique_strides, (void const *)unique_strides.data(), unique_strides.size() * sizeof(size_t),
                            hipMemcpyHostToDevice));

        C.update_D2H(); // We assume the most up-to-date data is on the device, but we need to do this part in core.
        T C_temp            = *(T *)C.host_data();
        *(T *)C.host_data() = T{0.0};

        if (C_prefactor == T{0.0}) {
            C_temp = T{0.0};
        } else {
            C_temp *= C_prefactor;
        }

        C.update_H2D();

        auto threads = gpu::block_size(A.dim(0) * unique_strides[0]), blocks = gpu::blocks(A.dim(0) * unique_strides[0]);

        detail::dot_kernel<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
            gpu_unique_strides, (DevDatatype<T> *)C.dev_data(), (DevDatatype<T> *)A.dev_data(), A.gpu_strides(),
            (DevDatatype<T> *)B.dev_data(), B.gpu_strides(), A.dim(0) * unique_strides[0], A.rank());

        gpu::stream_wait();

        C.update_D2H();
        *(T *)C.host_data() *= AB_prefactor;
        *(T *)C.host_data() += C_temp;
        C.update_H2D();

        hip_catch(hipFree((void *)gpu_unique_strides));
    }
#endif

    template <typename T>
    void execute_imp(T C_prefactor, pybind11::buffer &C, T AB_prefactor, pybind11::buffer const &A, pybind11::buffer const &B) const {
        pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
        if (A_info.ndim != B_info.ndim) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Tensor ranks do not match the indices!");
        }

        T *C_data = (T *)C_info.ptr;

        if (C_prefactor == T{0.0}) {
            *C_data = T{0.0};
        } else {
            *C_data *= C_prefactor;
        }

        *C_data = AB_prefactor * pybind11::cast<T>(einsums::python::detail::dot(A, B));
    }

  public:
    PyEinsumDotPlan() = delete;
    explicit PyEinsumDotPlan(PyEinsumGenericPlan const &plan_base);
    PyEinsumDotPlan(PyEinsumDotPlan const &) = default;
    ~PyEinsumDotPlan()                       = default;

#ifdef EINSUMS_COMPUTE_CODE
    virtual void execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                         python::PyGPUView const &A, python::PyGPUView const &B) const override;
#endif

    virtual void execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                         pybind11::buffer const &A, pybind11::buffer const &B) const override;
};

class EINSUMS_EXPORT PyEinsumDirectProductPlan : public PyEinsumGenericPlan {
  private:
#ifdef EINSUMS_COMPUTE_CODE
    template <typename T>
    void execute_imp_gpu(T C_prefactor, python::PyGPUView &C, T AB_prefactor, python::PyGPUView const &A,
                         python::PyGPUView const &B) const {
        if (A.rank() != B.rank() || A.rank() != C.rank() || A.rank() != this->_num_inds) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Tensor ranks do not match the indices!");
        }
        std::vector<size_t> index_strides(C.rank());
        size_t              elements = 1;
        size_t              rank     = A.rank();

        for (int i = A.rank() - 1; i >= 0; i--) {
            index_strides[i] = elements;
            elements *= A.dim(i);
        }

        size_t *gpu_index_strides;

        hip_catch(hipMalloc((void **)&gpu_index_strides, rank * sizeof(size_t)));

        hip_catch(hipMemcpy((void *)gpu_index_strides, (void const *)index_strides.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));

        auto threads = gpu::block_size(elements), blocks = gpu::blocks(elements);

        detail::direct_product_kernel<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
            gpu_index_strides, HipCast<DevDatatype<T>, T>::cast(C_prefactor), (DevDatatype<T> *)C.dev_data(), C.gpu_strides(),
            HipCast<DevDatatype<T>, T>::cast(AB_prefactor), (DevDatatype<T> *)A.dev_data(), A.gpu_strides(), (DevDatatype<T> *)B.dev_data(),
            B.gpu_strides(), elements, rank);

        gpu::stream_wait();

        hip_catch(hipFree((void *)gpu_index_strides));
    }
#endif

    template <typename T>
    void execute_imp(T C_prefactor, pybind11::buffer &C, T AB_prefactor, pybind11::buffer const &A, pybind11::buffer const &B) const {
        pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);
        if (A_info.ndim != B_info.ndim || A_info.ndim != C_info.ndim || A_info.ndim != this->_num_inds) {
            EINSUMS_THROW_EXCEPTION(num_argument_error, "Tensor ranks do not match the indices!");
        }
        std::vector<size_t> index_strides(C_info.ndim);
        size_t              elements = 1;
        size_t              rank     = A_info.ndim;

        for (int i = A_info.ndim - 1; i >= 0; i--) {
            index_strides[i] = elements;
            elements *= A_info.shape[i];
        }

        T       *C_data = (T *)C_info.ptr;
        T const *A_data = (T const *)A_info.ptr, *B_data = (T const *)B_info.ptr;

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < elements; sentinel++) {
            size_t quotient = sentinel;
            size_t A_index = 0, B_index = 0, C_index = 0;
            for (int i = 0; i < rank; i++) {
                size_t index = quotient / index_strides[i];
                quotient %= index_strides[i];

                A_index += (A_info.strides[i] / sizeof(T)) * index;
                B_index += (B_info.strides[i] / sizeof(T)) * index;
                C_index += (C_info.strides[i] / sizeof(T)) * index;
            }

            if (C_prefactor == T{0.0}) {
                C_data[C_index] = AB_prefactor * A_data[A_index] * B_data[B_index];
            } else {
                C_data[C_index] = C_prefactor * C_data[C_index] + AB_prefactor * A_data[A_index] * B_data[B_index];
            }
        }
    }

  public:
    PyEinsumDirectProductPlan() = delete;
    explicit PyEinsumDirectProductPlan(PyEinsumGenericPlan const &plan_base);
    PyEinsumDirectProductPlan(PyEinsumDirectProductPlan const &) = default;
    ~PyEinsumDirectProductPlan()                                 = default;

#ifdef EINSUMS_COMPUTE_CODE
    virtual void execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                         python::PyGPUView const &A, python::PyGPUView const &B) const override;
#endif

    virtual void execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                         pybind11::buffer const &A, pybind11::buffer const &B) const override;
};

class EINSUMS_EXPORT PyEinsumGerPlan : public PyEinsumGenericPlan {
  private:
    bool _swap_AB;

    std::vector<int> _CA_target_pos, _CB_target_pos;

#ifdef EINSUMS_COMPUTE_CODE
    template <typename T>
    void execute_imp_gpu(T C_prefactor, python::PyGPUView &C, T AB_prefactor, python::PyGPUView const &A,
                         python::PyGPUView const &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {

            size_t dC0 = 1, dC1 = 1;

            for (auto const i : _CA_target_pos) {
                dC0 *= C.dim(i);
            }

            for (auto const i : _CB_target_pos) {
                dC1 *= C.dim(i);
            }

            if (this->_swap_AB) {
                std::swap(dC0, dC1);
            }

            DevDatatype<T> *gpu_AB_prefactor;

            hip_catch(hipMalloc((void **)&gpu_AB_prefactor, sizeof(T)));
            hip_catch(hipMemcpy((void *)gpu_AB_prefactor, &AB_prefactor, sizeof(T), hipMemcpyHostToDevice));

            // Scale the matrix.
            if (C_prefactor != T{1.0}) {
                auto threads = gpu::block_size(C.size()), blocks = gpu::blocks(C.size());
                detail::scale_array<<<threads, blocks, 0, gpu::get_stream()>>>(HipCast<DevDatatype<T>, T>::cast(C_prefactor),
                                                                               (DevDatatype<T> *)C.dev_data(), C.size());
            }

            DevDatatype<T> const *gpu_A = (DevDatatype<T> *)A.dev_data(), *gpu_B = (DevDatatype<T> *)B.dev_data();

            if (this->_swap_AB) {
                std::swap(gpu_A, gpu_B);
            }

            if constexpr (std::is_same_v<T, float>) {
                hipblas_catch(hipblasSger(gpu::get_blas_handle(), dC1, dC0, gpu_AB_prefactor, gpu_B, 1, gpu_A, 1,
                                          (DevDatatype<T> *)C.dev_data(), dC1));
            } else if constexpr (std::is_same_v<T, double>) {
                hipblas_catch(hipblasDger(gpu::get_blas_handle(), dC1, dC0, gpu_AB_prefactor, gpu_B, 1, gpu_A, 1,
                                          (DevDatatype<T> *)C.dev_data(), dC1));
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                hipblas_catch(hipblasCgeru(gpu::get_blas_handle(), dC1, dC0, (hipblasComplex const *)gpu_AB_prefactor,
                                           (hipblasComplex const *)gpu_B, 1, (hipblasComplex const *)gpu_A, 1,
                                           (hipblasComplex *)C.dev_data(), dC1));
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                hipblas_catch(hipblasZgeru(gpu::get_blas_handle(), dC1, dC0, (hipblasDoubleComplex const *)gpu_AB_prefactor,
                                           (hipblasDoubleComplex const *)gpu_B, 1, (hipblasDoubleComplex const *)gpu_A, 1,
                                           (hipblasDoubleComplex *)C.dev_data(), dC1));
            }

            gpu::stream_wait();

            hip_catch(hipFree((void *)gpu_AB_prefactor));
        }
    }
#endif

    template <typename T>
    void execute_imp(T C_prefactor, pybind11::buffer &C, T AB_prefactor, pybind11::buffer const &A, pybind11::buffer const &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {
            pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);

            T const *A_data = (T const *)A_info.ptr, *B_data = (T const *)B_info.ptr;

            size_t dC0 = 1, dC1 = 1;

            for (auto const i : _CA_target_pos) {
                dC0 *= C_info.shape[i];
            }

            for (auto const i : _CB_target_pos) {
                dC1 *= C_info.shape[i];
            }

            if (this->_swap_AB) {
                std::swap(dC0, dC1);
            }

            T *C_data = (T *)C_info.ptr;

            if (C_prefactor == T{0.0}) {
                std::memset(C_data, 0, C_info.shape[0] * C_info.strides[0]);
            } else if (C_prefactor != T{1.0}) {
                EINSUMS_OMP_PARALLEL_FOR_SIMD
                for (size_t i = 0; i < C_info.shape[0] * C_info.strides[0]; i++) {
                    C_data[i] *= C_prefactor;
                }
            }

            if (this->_swap_AB) {
                std::swap(A_data, B_data);
            }

            einsums::blas::ger(dC0, dC1, AB_prefactor, A_data, 1, B_data, 1, C_data, dC1);
        }
    }

  public:
    PyEinsumGerPlan() = delete;
    explicit PyEinsumGerPlan(std::vector<int> const &CA_target_pos, std::vector<int> const &CB_target_pos, bool swap_AB,
                             PyEinsumGenericPlan const &plan_base);
    PyEinsumGerPlan(PyEinsumGerPlan const &) = default;
    ~PyEinsumGerPlan()                       = default;

#ifdef EINSUMS_COMPUTE_CODE
    virtual void execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                         python::PyGPUView const &A, python::PyGPUView const &B) const override;
#endif

    virtual void execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                         pybind11::buffer const &A, pybind11::buffer const &B) const override;
};

class EINSUMS_EXPORT PyEinsumGemvPlan : public PyEinsumGenericPlan {
  private:
    std::vector<int> _AC_pos, _A_link_pos, _B_link_pos;
    int              _A_target_last_ind, _A_link_last_ind, _B_link_last_ind, _C_target_last_ind;
    bool             _trans_A, _swap_AB;

#ifdef EINSUMS_COMPUTE_CODE
    template <typename T>
    void execute_imp_gpu(T C_prefactor, python::PyGPUView &C, T AB_prefactor, python::PyGPUView const &A,
                         python::PyGPUView const &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {
            size_t A_size = A.size(), B_size = B.size(), C_size = C.size();

            size_t dC = 1, dA0 = 1, dA1 = 1, dB = 1, sA0, sA1, sB, sC;

            for (int i = 0; i < _AC_pos.size(); i++) {
                dA0 *= C.dim(_AC_pos[i]);
            }
            for (int i = 0; i < _AC_pos.size(); i++) {
                dA1 *= A.dim(_A_link_pos[i]);
            }

            sA0 = A.stride(_A_target_last_ind) / sizeof(T);
            sA1 = A.stride(_A_link_last_ind) / sizeof(T);

            if (_trans_A) {
                std::swap(dA0, dA1);
                std::swap(sA0, sA1);
            }

            for (int i = 0; i < _B_link_pos.size(); i++) {
                dB *= B.dim(_B_link_pos[i]);
            }

            sB = B.stride(_B_link_last_ind) / sizeof(T);

            for (int i = 0; i < _AC_pos.size(); i++) {
                dC *= C.dim(_AC_pos[i]);
            }

            sC = C.stride(_C_target_last_ind) / sizeof(T);

            DevDatatype<T> *gpu_C_prefactor, *gpu_AB_prefactor;

            hip_catch(hipMalloc((void **)&gpu_AB_prefactor, sizeof(T)));
            hip_catch(hipMemcpy((void *)gpu_AB_prefactor, &AB_prefactor, sizeof(T), hipMemcpyHostToDevice));
            hip_catch(hipMalloc((void **)&gpu_C_prefactor, sizeof(T)));
            hip_catch(hipMemcpy((void *)gpu_C_prefactor, &AB_prefactor, sizeof(T), hipMemcpyHostToDevice));

            if constexpr (std::is_same_v<T, float>) {
                hipblas_catch(hipblasSgemv(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA1, dA0, gpu_AB_prefactor,
                                           (float const *)A.dev_data(), sA0, (float const *)B.dev_data(), sB, gpu_C_prefactor,
                                           (float *)C.dev_data(), sC));
            } else if constexpr (std::is_same_v<T, double>) {
                hipblas_catch(hipblasDgemv(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA1, dA0, gpu_AB_prefactor,
                                           (double const *)A.dev_data(), sA0, (double const *)B.dev_data(), sB, gpu_C_prefactor,
                                           (double *)C.dev_data(), sC));
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                hipblas_catch(hipblasCgemv(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA1, dA0,
                                           (hipblasComplex const *)gpu_AB_prefactor, (hipblasComplex const *)A.dev_data(), sA0,
                                           (hipblasComplex const *)B.dev_data(), sB, (hipblasComplex const *)gpu_C_prefactor,
                                           (hipblasComplex *)C.dev_data(), sC));
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                hipblas_catch(hipblasZgemv(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA1, dA0,
                                           (hipblasDoubleComplex const *)gpu_AB_prefactor, (hipblasDoubleComplex const *)A.dev_data(), sA0,
                                           (hipblasDoubleComplex const *)B.dev_data(), sB, (hipblasDoubleComplex const *)gpu_C_prefactor,
                                           (hipblasDoubleComplex *)C.dev_data(), sC));
            }

            gpu::stream_wait();

            hip_catch(hipFree((void *)gpu_AB_prefactor));
            hip_catch(hipFree((void *)gpu_C_prefactor));
        }
    }
#endif

    template <typename T>
    void execute_imp(T C_prefactor, pybind11::buffer &C, T AB_prefactor, pybind11::buffer const &A, pybind11::buffer const &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {
            pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);

            if (_swap_AB) {
                std::swap(A_info, B_info);
            }

            T const *A_data = (T const *)A_info.ptr, *B_data = (T const *)B_info.ptr;

            size_t dC = 1, dA0 = 1, dA1 = 1, dB = 1, sA0, sA1, sB, sC;

            for (int i = 0; i < _AC_pos.size(); i++) {
                assert(_AC_pos[i] < C_info.ndim);
                dA0 *= C_info.shape[_AC_pos[i]];
            }
            for (int i = 0; i < _AC_pos.size(); i++) {
                assert(_A_link_pos[i] < A_info.ndim);
                dA1 *= A_info.shape[_A_link_pos[i]];
            }

            assert(_A_target_last_ind < A_info.ndim);
            assert(_A_link_last_ind < A_info.ndim);

            sA0 = A_info.strides[_A_target_last_ind] / sizeof(T);
            sA1 = A_info.strides[_A_link_last_ind] / sizeof(T);

            if (_trans_A) {
                std::swap(dA0, dA1);
                std::swap(sA0, sA1);
            }

            for (int i = 0; i < _B_link_pos.size(); i++) {
                assert(_B_link_pos[i] < B_info.ndim);
                dB *= B_info.shape[_B_link_pos[i]];
            }

            sB = B_info.strides[_B_link_last_ind] / sizeof(T);

            for (int i = 0; i < _AC_pos.size(); i++) {
                assert(_AC_pos[i] < C_info.ndim);
                dC *= C_info.shape[_AC_pos[i]];
            }

            assert(_C_target_last_ind < C_info.ndim);
            sC = C_info.strides[_C_target_last_ind] / sizeof(T);

            T *C_data = (T *)C_info.ptr;

            einsums::blas::gemv((_trans_A) ? 'T' : 'N', dA0, dA1, AB_prefactor, A_data, sA0, B_data, sB, C_prefactor, C_data, sC);
        }
    }

  public:
    PyEinsumGemvPlan() = delete;
    explicit PyEinsumGemvPlan(std::vector<int> const &A_link_pos, std::vector<int> const &B_link_pos, std::vector<int> const &AC_pos,
                              int A_target_last_ind, int A_link_last_ind, int B_link_last_ind, int C_target_last_ind, bool trans_A,
                              bool swap_AB, PyEinsumGenericPlan const &plan_base);
    PyEinsumGemvPlan(PyEinsumGemvPlan const &) = default;
    ~PyEinsumGemvPlan()                        = default;

#ifdef EINSUMS_COMPUTE_CODE
    virtual void execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                         python::PyGPUView const &A, python::PyGPUView const &B) const override;
#endif

    virtual void execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                         pybind11::buffer const &A, pybind11::buffer const &B) const override;
};

class EINSUMS_EXPORT PyEinsumGemmPlan : public PyEinsumGenericPlan {
  private:
    std::vector<int> _AC_inds, _BC_inds, _A_link_inds, _B_link_inds;
    int              _A_target_last_ind, _A_link_last_ind, _B_target_last_ind, _B_link_last_ind, _CA_target_last_ind, _CB_target_last_ind;
    bool             _trans_A, _trans_B, _trans_C;

#ifdef EINSUMS_COMPUTE_CODE
    template <typename T>
    void execute_imp_gpu(T C_prefactor, python::PyGPUView &C, T AB_prefactor, python::PyGPUView const &A,
                         python::PyGPUView const &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {

            size_t dC0 = 1, dC1 = 1, dA0 = 1, dA1 = 1, dB0 = 1, dB1 = 1, sA0, sA1, sB0, sB1, sC0, sC1;

            for (int i = 0; i < _AC_inds.size(); i++) {
                dA0 *= C.dim(_AC_inds[i]);
            }
            for (int i = 0; i < _AC_inds.size(); i++) {
                dA1 *= A.dim(_A_link_inds[i]);
            }

            sA0 = A.stride(_A_target_last_ind) / sizeof(T);
            sA1 = A.stride(_A_link_last_ind) / sizeof(T);

            if (_trans_A) {
                std::swap(dA0, dA1);
                std::swap(sA0, sA1);
            }

            for (int i = 0; i < _BC_inds.size(); i++) {
                dB1 *= C.dim(_BC_inds[i]);
            }
            for (int i = 0; i < _BC_inds.size(); i++) {
                dB0 *= B.dim(_B_link_inds[i]);
            }

            sB1 = B.stride(_B_target_last_ind) / sizeof(T);
            sB0 = B.stride(_B_link_last_ind) / sizeof(T);

            if (_trans_B) {
                std::swap(dB0, dB1);
                std::swap(sB0, sB1);
            }

            for (int i = 0; i < _AC_inds.size(); i++) {
                dC0 *= C.dim(_AC_inds[i]);
            }

            for (int i = 0; i < _BC_inds.size(); i++) {
                dC1 *= C.dim(_BC_inds[i]);
            }

            sC0 = C.stride(_CA_target_last_ind) / sizeof(T);
            sC1 = C.stride(_CB_target_last_ind) / sizeof(T);

            if (_trans_C) {
                std::swap(dC0, dC1);
                std::swap(sC0, sC1);
            }

            DevDatatype<T> *gpu_C_prefactor, *gpu_AB_prefactor;

            hip_catch(hipMalloc((void **)&gpu_AB_prefactor, sizeof(T)));
            hip_catch(hipMemcpy((void *)gpu_AB_prefactor, &AB_prefactor, sizeof(T), hipMemcpyHostToDevice));
            hip_catch(hipMalloc((void **)&gpu_C_prefactor, sizeof(T)));
            hip_catch(hipMemcpy((void *)gpu_C_prefactor, &AB_prefactor, sizeof(T), hipMemcpyHostToDevice));

            if (_trans_C) {
                if constexpr (std::is_same_v<T, float>) {
                    hipblas_catch(hipblasSgemm(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (!_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA0, dB1, dA1, gpu_AB_prefactor,
                                               (float const *)A.dev_data(), sA0, (float const *)B.dev_data(), sB0, gpu_C_prefactor,
                                               (float *)C.dev_data(), sC0));
                } else if constexpr (std::is_same_v<T, double>) {
                    hipblas_catch(hipblasDgemm(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (!_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA0, dB1, dA1, gpu_AB_prefactor,
                                               (double const *)A.dev_data(), sA0, (double const *)B.dev_data(), sB0, gpu_C_prefactor,
                                               (double *)C.dev_data(), sC0));
                } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                    hipblas_catch(hipblasCgemm(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (!_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA0, dB1, dA1,
                                               (hipblasComplex const *)gpu_AB_prefactor, (hipblasComplex const *)A.dev_data(), sA0,
                                               (hipblasComplex const *)B.dev_data(), sB0, (hipblasComplex const *)gpu_C_prefactor,
                                               (hipblasComplex *)C.dev_data(), sC0));
                } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    hipblas_catch(hipblasZgemm(gpu::get_blas_handle(), (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (!_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dA0, dB1, dA1,
                                               (hipblasDoubleComplex const *)gpu_AB_prefactor, (hipblasDoubleComplex const *)A.dev_data(),
                                               sA0, (hipblasDoubleComplex const *)B.dev_data(), sB0,
                                               (hipblasDoubleComplex const *)gpu_C_prefactor, (hipblasDoubleComplex *)C.dev_data(), sC0));
                }
            } else {
                if constexpr (std::is_same_v<T, float>) {
                    hipblas_catch(hipblasSgemm(gpu::get_blas_handle(), (_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dB1, dA0, dA1, gpu_AB_prefactor,
                                               (float const *)B.dev_data(), sB0, (float const *)A.dev_data(), sA0, gpu_C_prefactor,
                                               (float *)C.dev_data(), sC0));
                } else if constexpr (std::is_same_v<T, double>) {
                    hipblas_catch(hipblasDgemm(gpu::get_blas_handle(), (_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dB1, dA0, dA1, gpu_AB_prefactor,
                                               (double const *)B.dev_data(), sB0, (double const *)A.dev_data(), sA0, gpu_C_prefactor,
                                               (double *)C.dev_data(), sC0));
                } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                    hipblas_catch(hipblasCgemm(gpu::get_blas_handle(), (_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dB1, dA0, dA1, (hipblasComplex *)gpu_AB_prefactor,
                                               (hipblasComplex const *)B.dev_data(), sB0, (hipblasComplex const *)A.dev_data(), sA0,
                                               (hipblasComplex *)gpu_C_prefactor, (hipblasComplex *)C.dev_data(), sC0));
                } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    hipblas_catch(hipblasZgemm(gpu::get_blas_handle(), (_trans_B) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               (_trans_A) ? HIPBLAS_OP_T : HIPBLAS_OP_N, dB1, dA0, dA1,
                                               (hipblasDoubleComplex const *)gpu_AB_prefactor, (hipblasDoubleComplex const *)B.dev_data(),
                                               sB0, (hipblasDoubleComplex const *)A.dev_data(), sA0,
                                               (hipblasDoubleComplex const *)gpu_C_prefactor, (hipblasDoubleComplex *)C.dev_data(), sC0));
                }
            }

            gpu::stream_wait();

            hip_catch(hipFree((void *)gpu_AB_prefactor));
            hip_catch(hipFree((void *)gpu_C_prefactor));
        }
    }
#endif

    template <typename T>
    void execute_imp(T C_prefactor, pybind11::buffer &C, T AB_prefactor, pybind11::buffer const &A, pybind11::buffer const &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {
            pybind11::buffer_info C_info = C.request(true), A_info = A.request(false), B_info = B.request(false);

            T const *A_data = (T const *)A_info.ptr, *B_data = (T const *)B_info.ptr;

            size_t dC0 = 1, dC1 = 1, dA0 = 1, dA1 = 1, dB0 = 1, dB1 = 1, sA0, sA1, sB0, sB1, sC0, sC1;

            for (int i = 0; i < _AC_inds.size(); i++) {
                dA0 *= C_info.shape[_AC_inds[i]];
            }
            for (int i = 0; i < _AC_inds.size(); i++) {
                dA1 *= A_info.shape[_A_link_inds[i]];
            }

            sA0 = A_info.strides[_A_target_last_ind] / sizeof(T);
            sA1 = A_info.strides[_A_link_last_ind] / sizeof(T);

            if (_trans_A) {
                std::swap(dA0, dA1);
                std::swap(sA0, sA1);
            }

            for (int i = 0; i < _BC_inds.size(); i++) {
                dB1 *= C_info.shape[_BC_inds[i]];
            }
            for (int i = 0; i < _BC_inds.size(); i++) {
                dB0 *= B_info.shape[_B_link_inds[i]];
            }

            sB1 = B_info.strides[_B_target_last_ind] / sizeof(T);
            sB0 = B_info.strides[_B_link_last_ind] / sizeof(T);

            if (_trans_B) {
                std::swap(dB0, dB1);
                std::swap(sB0, sB1);
            }

            for (int i = 0; i < _AC_inds.size(); i++) {
                dC0 *= C_info.shape[_AC_inds[i]];
            }

            for (int i = 0; i < _BC_inds.size(); i++) {
                dC1 *= C_info.shape[_BC_inds[i]];
            }

            sC0 = C_info.strides[_CA_target_last_ind] / sizeof(T);
            sC1 = C_info.strides[_CB_target_last_ind] / sizeof(T);

            if (_trans_C) {
                std::swap(dC0, dC1);
                std::swap(sC0, sC1);
            }

            T *C_data = (T *)C_info.ptr;

            if (!_trans_C) {
                einsums::blas::gemm((_trans_A) ? 'T' : 'N', (_trans_B) ? 'T' : 'N', dA0, dB1, dA1, AB_prefactor, A_data, sA0, B_data, sB0,
                                    C_prefactor, C_data, sC0);
            } else {
                einsums::blas::gemm((!_trans_B) ? 'T' : 'N', (!_trans_A) ? 'T' : 'N', dB1, dA0, dA1, AB_prefactor, B_data, sB0, A_data, sA0,
                                    C_prefactor, C_data, sC0);
            }
        }
    }

  public:
    PyEinsumGemmPlan() = delete;
    explicit PyEinsumGemmPlan(std::vector<int> const &A_link_inds, std::vector<int> const &B_link_inds, std::vector<int> const &AC_inds,
                              std::vector<int> const &BC_inds, int A_target_last_ind, int A_link_last_ind, int B_target_last_ind,
                              int B_link_last_ind, int CA_target_last_ind, int CB_target_last_ind, bool trans_A, bool trans_B, bool trans_C,
                              PyEinsumGenericPlan const &plan_base);
    PyEinsumGemmPlan(PyEinsumGemmPlan const &) = default;
    ~PyEinsumGemmPlan()                        = default;

#ifdef EINSUMS_COMPUTE_CODE
    virtual void execute(pybind11::object const &C_prefactor, python::PyGPUView &C, pybind11::object const &AB_prefactor,
                         python::PyGPUView const &A, python::PyGPUView const &B) const override;
#endif

    virtual void execute(pybind11::object const &C_prefactor, pybind11::buffer &C, pybind11::object const &AB_prefactor,
                         pybind11::buffer const &A, pybind11::buffer const &B) const override;
};

EINSUMS_EXPORT std::shared_ptr<PyEinsumGenericPlan> compile_plan(std::string C_indices, std::string A_indices, std::string B_indices);

} // namespace einsums::tensor_algebra