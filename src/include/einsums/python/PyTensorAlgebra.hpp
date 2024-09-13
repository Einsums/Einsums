#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

#include "einsums/Blas.hpp"
#include "einsums/Exception.hpp"
#include "einsums/Python.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorBases.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"

#    include "einsums/tensor_algebra_backends/GPUTensorAlgebra.hpp"

#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace einsums::tensor_algebra {

namespace detail {
EINSUMS_EXPORT std::vector<size_t> get_dim_ranges_for_many(const pybind11::array &C, const std::vector<int> &C_perm,
                                                           const pybind11::array &A, const std::vector<int> &A_perm,
                                                           const pybind11::array &B, const std::vector<int> &B_perm, int unique_indices);

EINSUMS_EXPORT std::string intersect(const std::string &st1, const std::string &st2);

template <typename T>
std::vector<T> intersect(const std::vector<T> &vec1, const std::vector<T> &vec2) {
    std::vector<T> out;

    for (int i = 0; i < vec1.size(); i++) {
        for (int j = 0; j < vec2.size(); j++) {
            if (vec1[i] == vec2[j]) {
                out.push_back(vec1[i]);
            }
        }
    }
    return out;
}

#ifdef __HIP__

template <typename DataType>
__global__ void einsum_generic_algorithm_gpu(const size_t *__restrict__ unique_strides, const size_t *__restrict__ C_index_strides,
                                             const int *__restrict__ C_index_table, const int *__restrict__ A_index_table,
                                             const int *__restrict__ B_index_table, const DataType C_prefactor, DataType *__restrict__ C,
                                             const size_t *__restrict__ C_stride, const size_t *__restrict__ C_stride_unique,
                                             const DataType AB_prefactor, const DataType *__restrict__ A,
                                             const size_t *__restrict__ A_stride, const size_t *__restrict__ A_stride_unique,
                                             const DataType *__restrict__ B, const size_t *__restrict__ B_stride,
                                             const size_t *__restrict__ B_stride_unique, size_t max_index, size_t C_size, size_t C_rank,
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

        einsums::gpu::atomicAdd_wrap(C + C_sentinel, AB_prefactor * A[A_sentinel] * B[B_sentinel]);
    }
}

// When we will only see a certain element once, we can ignore atomicity for a speedup.
template <typename DataType>
__global__ void einsum_generic_algorithm_direct_product_gpu(
    const size_t *__restrict__ unique_strides, const size_t *__restrict__ C_index_strides, const int *__restrict__ C_index_table,
    const int *__restrict__ A_index_table, const int *__restrict__ B_index_table, const DataType    C_prefactor, DataType *__restrict__ C,
    const size_t *__restrict__ C_stride, const size_t *__restrict__ C_stride_unique, const DataType AB_prefactor,
    const DataType *__restrict__ A, const size_t *__restrict__ A_stride, const size_t *__restrict__ A_stride_unique,
    const DataType *__restrict__ B, const size_t *__restrict__ B_stride, const size_t *__restrict__ B_stride_unique, size_t max_index,
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
            C[C_sentinel] = AB_prefactor * A[A_sentinel] * B[B_sentinel];
        } else {
            C[C_sentinel] = C_prefactor * C[C_sentinel] + AB_prefactor * A[A_sentinel] * B[B_sentinel];
        }
    }
}

/**
 * Compute kernel that runs when C has a rank of zero. There are some optimizations that can be made in this case.
 */
template <typename DataType>
__global__ void einsum_generic_zero_rank_gpu(const size_t *__restrict__ unique_strides, const int *__restrict__ A_index_table,
                                             const int *__restrict__ B_index_table, DataType *__restrict__ C, const DataType AB_prefactor,
                                             const DataType *__restrict__ A, const size_t *__restrict__ A_stride,
                                             const size_t *__restrict__ A_stride_unique, const DataType *__restrict__ B,
                                             const size_t *__restrict__ B_stride, const size_t *__restrict__ B_stride_unique,
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

        value = value + A[A_sentinel] * B[B_sentinel];
    }

    atomicAdd_wrap(C, AB_prefactor * value);
}

template <typename T>
__global__ void dot_kernel(const size_t *unique_strides, T *__restrict__ C, const T *__restrict__ A, const size_t *__restrict__ A_strides,
                           const T *__restrict__ B, const size_t *__restrict__ B_strides, size_t max_index, size_t rank) {
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

        value = value + A[A_sentinel] * B[B_sentinel];
    }

    atomicAdd_wrap(C, value);
}

template <typename DataType>
__global__ void direct_product_kernel(const size_t *__restrict__ index_strides, const DataType C_prefactor, DataType *__restrict__ C,
                                      const size_t *__restrict__ C_stride, const DataType      AB_prefactor, const DataType *__restrict__ A,
                                      const size_t *__restrict__ A_stride, const DataType *__restrict__ B,
                                      const size_t *__restrict__ B_stride, size_t max_index, size_t rank) {
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
            C[C_sentinel] = AB_prefactor * A[A_sentinel] * B[B_sentinel];
        } else {
            C[C_sentinel] = C_prefactor * C[C_sentinel] + AB_prefactor * A[A_sentinel] * B[B_sentinel];
        }
    }
}

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
            array[curr_index] = factor * array[curr_index];
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
class PyEinsumGenericPlan {
  protected:
    std::vector<int>                    _C_permute, _A_permute, _B_permute;
    int                                 _num_inds;
    einsums::python::detail::PyPlanUnit _unit;

    template <typename T>
    void execute_generic(T C_prefactor, pybind11::array &C, T AB_prefactor, const pybind11::array &A, const pybind11::array &B) const {
        using namespace einsums::tensor_algebra::detail;
        if (_C_permute.size() != C.ndim() || _A_permute.size() != A.ndim() || _B_permute.size() != B.ndim()) {
            throw EINSUMSEXCEPTION("Tensor ranks do not match the indices!");
        }
#ifdef __HIP__
        if (_unit == einsums::python::detail::GPU_MAP || _unit == einsums::python::detail::GPU_COPY) {
            T       *C_data = (T *)C.mutable_unchecked<T>().mutable_data();
            const T *A_data = (const T *)A.unchecked<T>().data(), *B_data = (const T *)B.unchecked<T>().data();

            size_t C_size = (size_t)C.shape(0) * (size_t)C.strides(0), A_size = (size_t)A.shape(0) * (size_t)A.strides(0),
                   B_size = (size_t)B.shape(0) * (size_t)B.strides(0);

            bool direct_product_swap = (_A_permute.size() == _B_permute.size()) && (_A_permute.size() == _C_permute.size()) &&
                                       (intersect(_A_permute, _B_permute).size() == _A_permute.size()) &&
                                       (intersect(_A_permute, _C_permute).size() == _A_permute.size()) &&
                                       (intersect(_B_permute, _C_permute).size() == _A_permute.size());
            std::vector<size_t> unique_dims = get_dim_ranges_for_many(C, _C_permute, A, _A_permute, B, _B_permute, _num_inds);

            std::vector<size_t> unique_strides, C_index_strides, C_dims, A_unique_stride, B_unique_stride, C_unique_stride;

            for (int i = 0; i < _C_permute.size(); i++) {
                C_dims.push_back(C.shape(i));
            }

            dims_to_strides(unique_dims, unique_strides);
            dims_to_strides(C_dims, C_index_strides);

            A_unique_stride.resize(unique_strides.size());
            B_unique_stride.resize(unique_strides.size());
            C_unique_stride.resize(unique_strides.size());

            for (int i = 0; i < A_unique_stride.size(); i++) {
                bool found = false;
                for (int j = 0; j < _A_permute.size(); j++) {
                    if (_A_permute[j] == i) {
                        A_unique_stride[i] = A.strides(j);
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
                        B_unique_stride[i] = B.strides(j);
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
                        C_unique_stride[i] = C.strides(j);
                        found              = true;
                        break;
                    }
                }
                if (!found) {
                    C_unique_stride[i] = 0;
                }
            }

            __device_ptr__ int    *gpu_C_index_table, *gpu_A_index_table, *gpu_B_index_table;
            __device_ptr__ size_t *gpu_unique_strides, *gpu_C_index_strides;

            __device_ptr__ size_t *gpu_C_dims, *gpu_C_stride, *gpu_A_stride, *gpu_B_stride, *gpu_A_unique_stride, *gpu_B_unique_stride,
                *gpu_C_unique_stride;

            __device_ptr__ DevDatatype<T> *gpu_C, *gpu_A, *gpu_B;

            if (_C_permute.size() != 0) {
                gpu::hip_catch(hipMalloc((void **)&gpu_C_dims, _C_permute.size() * sizeof(size_t)));
                gpu::hip_catch(hipMalloc((void **)&gpu_C_stride, _C_permute.size() * sizeof(size_t)));
                gpu::hip_catch(hipMalloc((void **)&gpu_C_unique_stride, unique_strides.size() * sizeof(size_t)));
            }
            gpu::hip_catch(hipMalloc((void **)&gpu_A_stride, _A_permute.size() * sizeof(size_t)));
            gpu::hip_catch(hipMalloc((void **)&gpu_B_stride, _B_permute.size() * sizeof(size_t)));
            gpu::hip_catch(hipMalloc((void **)&gpu_A_unique_stride, unique_strides.size() * sizeof(size_t)));
            gpu::hip_catch(hipMalloc((void **)&gpu_B_unique_stride, unique_strides.size() * sizeof(size_t)));

            gpu::hip_catch(hipMalloc((void **)&gpu_unique_strides, _num_inds * sizeof(size_t)));
            if (_C_permute.size() != 0) {
                gpu::hip_catch(hipMalloc((void **)&gpu_C_index_strides, _C_permute.size() * sizeof(size_t)));

                gpu::hip_catch(hipMalloc((void **)&gpu_C_index_table, _C_permute.size() * sizeof(int)));
            }
            gpu::hip_catch(hipMalloc((void **)&gpu_A_index_table, _A_permute.size() * sizeof(int)));
            gpu::hip_catch(hipMalloc((void **)&gpu_B_index_table, _B_permute.size() * sizeof(int)));

            gpu::hip_catch(hipMemcpy((void *)gpu_A_index_table, (const void *)_A_permute.data(), _A_permute.size() * sizeof(int),
                                     hipMemcpyHostToDevice));
            gpu::hip_catch(hipMemcpy((void *)gpu_B_index_table, (const void *)_B_permute.data(), _B_permute.size() * sizeof(int),
                                     hipMemcpyHostToDevice));
            gpu::hip_catch(hipMemcpy((void *)gpu_A_unique_stride, (const void *)A_unique_stride.data(),
                                     A_unique_stride.size() * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(hipMemcpy((void *)gpu_B_unique_stride, (const void *)B_unique_stride.data(),
                                     B_unique_stride.size() * sizeof(size_t), hipMemcpyHostToDevice));

            if (_C_permute.size() != 0) {
                gpu::hip_catch(hipMemcpy((void *)gpu_C_index_table, (const void *)_C_permute.data(), _C_permute.size() * sizeof(int),
                                         hipMemcpyHostToDevice));
                gpu::hip_catch(hipMemcpy((void *)gpu_C_unique_stride, (const void *)C_unique_stride.data(),
                                         C_unique_stride.size() * sizeof(size_t), hipMemcpyHostToDevice));
            }

            gpu::hip_catch(hipMemcpy((void *)gpu_unique_strides, (const void *)unique_strides.data(),
                                     unique_strides.size() * sizeof(size_t), hipMemcpyHostToDevice));

            gpu::hip_catch(
                hipMemcpy((void *)gpu_A_stride, (const void *)A.strides(), _A_permute.size() * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(
                hipMemcpy((void *)gpu_B_stride, (const void *)B.strides(), _B_permute.size() * sizeof(size_t), hipMemcpyHostToDevice));
            if (_C_permute.size() != 0) {
                gpu::hip_catch(
                    hipMemcpy((void *)gpu_C_stride, (const void *)C.strides(), _C_permute.size() * sizeof(size_t), hipMemcpyHostToDevice));
                gpu::hip_catch(
                    hipMemcpy((void *)gpu_C_dims, (const void *)C_dims.data(), _C_permute.size() * sizeof(size_t), hipMemcpyHostToDevice));
                gpu::hip_catch(hipMemcpy((void *)gpu_C_index_strides, (const void *)C_index_strides.data(),
                                         _C_permute.size() * sizeof(size_t), hipMemcpyHostToDevice));
            }

            if (_unit == python::detail::GPU_COPY) {
                // strides() is already multiplied by the item size.
                gpu::hip_catch(hipMalloc((void **)&gpu_A, A.shape(0) * A.strides(0)));
                gpu::hip_catch(hipMalloc((void **)&gpu_B, B.shape(0) * B.strides(0)));

                gpu::hip_catch(hipMemcpy((void *)gpu_A, (const void *)A_data, A.shape(0) * A.strides(0), hipMemcpyHostToDevice));
                gpu::hip_catch(hipMemcpy((void *)gpu_B, (const void *)B_data, B.shape(0) * B.strides(0), hipMemcpyHostToDevice));

                if (_C_permute.size() != 0) {
                    gpu::hip_catch(hipMalloc((void **)&gpu_C, C.shape(0) * C.strides(0)));

                    gpu::hip_catch(hipMemcpy((void *)gpu_C, (void *)C_data, C.shape(0) * C.strides(0), hipMemcpyHostToDevice));
                }
            } else {
                gpu::hip_catch(hipHostRegister((void *)A_data, A.shape(0) * A.strides(0), hipHostRegisterDefault));
                gpu::hip_catch(hipHostRegister((void *)B_data, B.shape(0) * B.strides(0), hipHostRegisterDefault));

                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_A, (void *)A_data, 0));
                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_B, (void *)B_data, 0));

                if (_C_permute.size() != 0) {
                    gpu::hip_catch(hipHostRegister((void *)C_data, C.shape(0) * C.strides(0), hipHostRegisterDefault));

                    gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_C, (void *)C_data, 0));
                }
            }

            auto threads = gpu::block_size(unique_dims[0] * unique_strides[0]), blocks = gpu::blocks(unique_dims[0] * unique_strides[0]);

            if (_C_permute.size() != 0) {
                if (!direct_product_swap) {
                    detail::einsum_generic_algorithm_gpu<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                        gpu_unique_strides, gpu_C_index_strides, gpu_C_index_table, gpu_A_index_table, gpu_B_index_table,
                        gpu::HipCast<DevDatatype<T>, T>::cast(C_prefactor), gpu_C, gpu_C_stride, gpu_C_unique_stride,
                        gpu::HipCast<DevDatatype<T>, T>::cast(AB_prefactor), gpu_A, gpu_A_stride, gpu_A_unique_stride, gpu_B, gpu_B_stride,
                        gpu_B_unique_stride, unique_dims[0] * unique_strides[0], C.size(), C.ndim(), A.ndim(), B.ndim(),
                        unique_dims.size());
                } else {
                    detail::einsum_generic_algorithm_direct_product_gpu<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                        gpu_unique_strides, gpu_C_index_strides, gpu_C_index_table, gpu_A_index_table, gpu_B_index_table,
                        gpu::HipCast<DevDatatype<T>, T>::cast(C_prefactor), gpu_C, gpu_C_stride, gpu_C_unique_stride,
                        gpu::HipCast<DevDatatype<T>, T>::cast(AB_prefactor), gpu_A, gpu_A_stride, gpu_A_unique_stride, gpu_B, gpu_B_stride,
                        gpu_B_unique_stride, unique_dims[0] * unique_strides[0], C.size(), C.ndim(), A.ndim(), B.ndim(),
                        unique_dims.size());
                }

                gpu::stream_wait();

                gpu::hip_catch(hipFree((void *)gpu_C_dims));
                gpu::hip_catch(hipFree((void *)gpu_C_stride));
                gpu::hip_catch(hipFree((void *)gpu_C_unique_stride));
                gpu::hip_catch(hipFree((void *)gpu_C_index_table));
                gpu::hip_catch(hipFree((void *)gpu_C_index_strides));

                if (_unit == python::detail::GPU_COPY) {
                    gpu::hip_catch(hipMemcpy((void *)C_data, (const void *)gpu_C, C.shape(0) * C.strides(0), hipMemcpyDeviceToHost));
                    gpu::hip_catch(hipFree((void *)gpu_C));
                } else {
                    gpu::hip_catch(hipHostUnregister((void *)C_data));
                }
            } else {
                T &C_val = *(T *)C.mutable_data();

                if (C_prefactor == T{0.0}) {
                    C_val = T{0.0};
                } else {
                    C_val *= C_prefactor;
                }

                gpu::hip_catch(hipMalloc((void **)&gpu_C, sizeof(T)));

                gpu::hip_catch(hipMemcpy((void *)gpu_C, &C_val, sizeof(T), hipMemcpyHostToDevice));

                detail::einsum_generic_zero_rank_gpu<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                    gpu_unique_strides, gpu_A_index_table, gpu_B_index_table, gpu_C, gpu::HipCast<DevDatatype<T>, T>::cast(AB_prefactor),
                    gpu_A, gpu_A_stride, gpu_A_unique_stride, gpu_B, gpu_B_stride, gpu_B_unique_stride, unique_dims[0] * unique_strides[0],
                    _A_permute.size(), _B_permute.size(), _num_inds);

                gpu::stream_wait();

                gpu::hip_catch(hipMemcpy((void *)&C_val, (void *)gpu_C, sizeof(T), hipMemcpyDeviceToHost));

                gpu::hip_catch(hipFree((void *)gpu_C));
            }

            if (_unit == python::detail::GPU_COPY) {
                gpu::hip_catch(hipFree((void *)gpu_A));
                gpu::hip_catch(hipFree((void *)gpu_B));
            } else {
                gpu::hip_catch(hipHostUnregister((void *)A_data));
                gpu::hip_catch(hipHostUnregister((void *)B_data));
            }

            gpu::hip_catch(hipFree((void *)gpu_A_index_table));
            gpu::hip_catch(hipFree((void *)gpu_B_index_table));
            gpu::hip_catch(hipFree((void *)gpu_unique_strides));
            gpu::hip_catch(hipFree((void *)gpu_A_stride));
            gpu::hip_catch(hipFree((void *)gpu_B_stride));
            gpu::hip_catch(hipFree((void *)gpu_A_unique_stride));
            gpu::hip_catch(hipFree((void *)gpu_B_unique_stride));
        } else {
#endif
            std::vector<size_t> unique_dims = detail::get_dim_ranges_for_many(C, _C_permute, A, _A_permute, B, _B_permute, _num_inds);
            std::vector<size_t> unique_strides, C_index_strides, C_dims(_C_permute.size());

            dims_to_strides(unique_dims, unique_strides);

            T       *C_data = (T *)C.mutable_unchecked<T>().mutable_data();
            const T *A_data = (const T *)A.unchecked<T>().data();
            const T *B_data = (const T *)B.unchecked<T>().data();

            for (int i = 0; i < _C_permute.size(); i++) {
                C_dims[i] = C.shape(i);
            }

            dims_to_strides(C_dims, C_index_strides);

            if (C_prefactor == T{0.0}) {
                EINSUMS_OMP_PARALLEL_FOR
                for (size_t sentinel = 0; sentinel < C_dims[0] * C_index_strides[0]; sentinel++) {
                    size_t quotient = sentinel;
                    size_t C_index  = 0;
                    for (int i = 0; i < _C_permute.size(); i++) {
                        C_index += (C.strides(i) / sizeof(T)) * (quotient / C_index_strides[i]);
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
                        C_index += (C.strides(i) / sizeof(T)) * (quotient / C_index_strides[i]);
                        quotient %= C_index_strides[i];
                    }

                    C_data[C_index] *= C_prefactor;
                }
            }

            EINSUMS_OMP_PARALLEL_FOR
            for (size_t sentinel = 0; sentinel < unique_dims[0] * unique_strides[0]; sentinel++) {
                size_t                           quotient = sentinel;
                thread_local std::vector<size_t> A_index(_A_permute.size()), B_index(_B_permute.size()), C_index(_C_permute.size()),
                    unique_index(unique_dims.size());

                for (int i = 0; i < unique_dims.size(); i++) {
                    unique_index[i] = quotient / unique_strides[i];
                    quotient %= unique_strides[i];
                }

                size_t A_ord = 0, B_ord = 0, C_ord = 0;

                for (int i = 0; i < _A_permute.size(); i++) {
                    A_ord += (A.strides(i) / sizeof(T)) * unique_index[_A_permute[i]];
                }

                for (int i = 0; i < _B_permute.size(); i++) {
                    B_ord += (B.strides(i) / sizeof(T)) * unique_index[_B_permute[i]];
                }

                for (int i = 0; i < _C_permute.size(); i++) {
                    C_ord += (C.strides(i) / sizeof(T)) * unique_index[_C_permute[i]];
                }

                C_data[C_ord] += AB_prefactor * A_data[A_ord] * B_data[B_ord];
            }
#ifdef __HIP__
        }
#endif
    }

  public:
    PyEinsumGenericPlan() = delete;
    PyEinsumGenericPlan(int num_inds, std::vector<int> C_permute, std::vector<int> A_permute, std::vector<int> B_permute,
                        python::detail::PyPlanUnit unit = python::detail::CPU);
    PyEinsumGenericPlan(const PyEinsumGenericPlan &) = default;
    ~PyEinsumGenericPlan()                           = default;

    virtual void execute(const pybind11::object &C_prefactor, pybind11::array &C, const pybind11::object &AB_prefactor,
                         const pybind11::array &A, const pybind11::array &B) const;
};

class PyEinsumDotPlan : public PyEinsumGenericPlan {
  private:
    template <typename T>
    void execute_imp(T C_prefactor, pybind11::array &C, T AB_prefactor, const pybind11::array &A, const pybind11::array &B) const {
        if (A.ndim() != B.ndim()) {
            throw EINSUMSEXCEPTION("Tensor ranks do not match the indices!");
        }

        std::vector<size_t> unique_strides(A.ndim());
        size_t              prod = 1;

        for (int i = A.ndim() - 1; i >= 0; i--) {
            unique_strides[i] = prod;
            prod *= A.shape(i);
        }

#ifdef __HIP__
        if (this->_unit == einsums::python::detail::GPU_MAP || this->_unit == einsums::python::detail::GPU_COPY) {
            T       *C_data = (T *)C.mutable_unchecked<T>().mutable_data();
            const T *A_data = (const T *)A.unchecked<T>().data(), *B_data = (const T *)B.unchecked<T>().data();

            size_t C_size = C.shape(0) * C.strides(0), A_size = (size_t)A.shape(0) * (size_t)A.strides(0),
                   B_size = B.shape(0) * B.strides(0);

            __device_ptr__ size_t *gpu_unique_strides, *gpu_A_strides, *gpu_B_strides;
            __device_ptr__ DevDatatype<T> *gpu_A, *gpu_B, *gpu_C;

            gpu::hip_catch(hipMalloc((void **)&gpu_unique_strides, unique_strides.size() * sizeof(size_t)));
            gpu::hip_catch(hipMemcpy((void *)gpu_unique_strides, (const void *)unique_strides.data(),
                                     unique_strides.size() * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(hipMalloc((void **)&gpu_A_strides, A.ndim() * sizeof(size_t)));
            gpu::hip_catch(hipMemcpy((void *)gpu_A_strides, (const void *)A.strides(), A.ndim() * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(hipMalloc((void **)&gpu_B_strides, B.ndim() * sizeof(size_t)));
            gpu::hip_catch(hipMemcpy((void *)gpu_B_strides, (const void *)B.strides(), B.ndim() * sizeof(size_t), hipMemcpyHostToDevice));

            gpu::hip_catch(hipMalloc((void **)&gpu_C, sizeof(DevDatatype<T>)));

            if (this->_unit == einsums::python::detail::GPU_MAP) {
                gpu::hip_catch(hipHostRegister((void *)A_data, A_size, hipHostRegisterDefault));
                gpu::hip_catch(hipHostRegister((void *)B_data, B_size, hipHostRegisterDefault));
                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_A, (void *)A_data, 0));
                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_B, (void *)B_data, 0));
            } else {
                gpu_A = nullptr;
                gpu_B = nullptr;
                gpu::hip_catch(hipMalloc((void **)&gpu_A, A_size));
                gpu::hip_catch(hipMalloc((void **)&gpu_B, B_size));

                gpu::hip_catch(hipMemcpy((void *)gpu_A, (const void *)A_data, A_size, hipMemcpyHostToDevice));
                gpu::hip_catch(hipMemcpy((void *)gpu_B, (const void *)B_data, B_size, hipMemcpyHostToDevice));
            }

            auto threads = gpu::block_size(A.shape(0) * unique_strides[0]), blocks = gpu::blocks(A.shape(0) * unique_strides[0]);

            detail::dot_kernel<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                gpu_unique_strides, gpu_C, gpu_A, gpu_A_strides, gpu_B, gpu_B_strides, A.shape(0) * unique_strides[0], A.ndim());

            gpu::stream_wait();

            gpu::hip_catch(hipFree((void *)gpu_unique_strides));
            gpu::hip_catch(hipFree((void *)gpu_A_strides));
            gpu::hip_catch(hipFree((void *)gpu_B_strides));

            if (this->_unit == einsums::python::detail::GPU_MAP) {
                gpu::hip_catch(hipHostUnregister((void *)A_data));
                gpu::hip_catch(hipHostUnregister((void *)B_data));
            } else {
                gpu::hip_catch(hipFree((void *)gpu_A));
                gpu::hip_catch(hipFree((void *)gpu_B));
            }

            T C_temp;

            gpu::hip_catch(hipMemcpy((void *)&C_temp, (const void *)gpu_C, sizeof(T), hipMemcpyDeviceToHost));

            if (C_prefactor == T{0.0}) {
                *C_data = T{0.0};
            } else {
                *C_data *= C_prefactor;
            }

            *C_data += AB_prefactor * C_temp;

            gpu::hip_catch(hipFree(gpu_C));

            gpu::stream_wait();
        } else {
#endif
            T       &C_data = *(T *)(C.mutable_unchecked<T>().mutable_data());
            const T *A_data = (const T *)(A.unchecked<T>().data());
            const T *B_data = (const T *)(B.unchecked<T>().data());

            if (C_prefactor == T{0.0}) {
                C_data = T{0.0};
            } else {
                C_data *= C_prefactor;
            }

#pragma omp parallel for simd reduction(+ : C_data)
            for (size_t sentinel = 0; sentinel < A.shape(0) * unique_strides[0]; sentinel++) {
                size_t quotient = sentinel;
                size_t A_index  = 0;
                size_t B_index  = 0;
                for (int i = 0; i < A.ndim(); i++) {
                    size_t unique_index = quotient / unique_strides[i];
                    quotient %= unique_strides[i];

                    A_index += (A.strides(i) / sizeof(T)) * unique_index;
                    B_index += (B.strides(i) / sizeof(T)) * unique_index;
                }

                C_data += AB_prefactor * A_data[A_index] * B_data[B_index];
            }
#ifdef __HIP__
        }
#endif
    }

  public:
    PyEinsumDotPlan() = delete;
    explicit PyEinsumDotPlan(const PyEinsumGenericPlan &plan_base);
    PyEinsumDotPlan(const PyEinsumDotPlan &) = default;
    ~PyEinsumDotPlan()                       = default;

    void execute(const pybind11::object &C_prefactor, pybind11::array &C, const pybind11::object &AB_prefactor, const pybind11::array &A,
                 const pybind11::array &B) const override;
};

class PyEinsumDirectProductPlan : public PyEinsumGenericPlan {
  private:
    template <typename T>
    void execute_imp(T C_prefactor, pybind11::array &C, T AB_prefactor, const pybind11::array &A, const pybind11::array &B) const {
        if (A.ndim() != B.ndim() || A.ndim() != C.ndim() || A.ndim() != this->_num_inds) {
            throw EINSUMSEXCEPTION("Tensor ranks do not match the indices!");
        }
        std::vector<size_t> index_strides(C.ndim());
        size_t              elements = 1;
        size_t              rank     = A.ndim();

        for (int i = A.ndim() - 1; i >= 0; i--) {
            index_strides[i] = elements;
            elements *= A.shape(i);
        }
#ifdef __HIP__
        if (this->_unit == einsums::python::detail::GPU_COPY || this->_unit == einsums::python::detail::GPU_MAP) {
            T       *C_data = (T *)C.mutable_unchecked<T>().mutable_data();
            const T *A_data = (const T *)A.unchecked<T>().data(), *B_data = (const T *)B.unchecked<T>().data();

            __device_ptr__ size_t *gpu_index_strides, *gpu_C_strides, *gpu_A_strides, *gpu_B_strides;
            __device_ptr__ DevDatatype<T> *gpu_C, *gpu_A, *gpu_B;

            gpu::hip_catch(hipMalloc((void **)&gpu_index_strides, rank * sizeof(size_t)));
            gpu::hip_catch(hipMalloc((void **)&gpu_C_strides, rank * sizeof(size_t)));
            gpu::hip_catch(hipMalloc((void **)&gpu_A_strides, rank * sizeof(size_t)));
            gpu::hip_catch(hipMalloc((void **)&gpu_B_strides, rank * sizeof(size_t)));

            gpu::hip_catch(
                hipMemcpy((void *)gpu_index_strides, (const void *)index_strides.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(hipMemcpy((void *)gpu_C_strides, (const void *)C.strides(), rank * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(hipMemcpy((void *)gpu_A_strides, (const void *)A.strides(), rank * sizeof(size_t), hipMemcpyHostToDevice));
            gpu::hip_catch(hipMemcpy((void *)gpu_B_strides, (const void *)B.strides(), rank * sizeof(size_t), hipMemcpyHostToDevice));

            if (this->_unit == einsums::python::detail::GPU_COPY) {
                // strides() is already multiplied by the element size.
                gpu::hip_catch(hipMalloc((void **)&gpu_C, C.shape(0) * C.strides(0)));
                gpu::hip_catch(hipMalloc((void **)&gpu_A, A.shape(0) * A.strides(0)));
                gpu::hip_catch(hipMalloc((void **)&gpu_B, B.shape(0) * B.strides(0)));

                gpu::hip_catch(hipMemcpy((void *)gpu_C, (const void *)C_data, C.shape(0) * C.strides(0), hipMemcpyHostToDevice));
                gpu::hip_catch(hipMemcpy((void *)gpu_A, (const void *)A_data, A.shape(0) * A.strides(0), hipMemcpyHostToDevice));
                gpu::hip_catch(hipMemcpy((void *)gpu_B, (const void *)B_data, B.shape(0) * B.strides(0), hipMemcpyHostToDevice));
            } else {
                gpu::hip_catch(hipHostRegister((void *)C_data, C.shape(0) * C.strides(0), hipHostRegisterDefault));
                gpu::hip_catch(hipHostRegister((void *)A_data, A.shape(0) * A.strides(0), hipHostRegisterDefault));
                gpu::hip_catch(hipHostRegister((void *)B_data, B.shape(0) * B.strides(0), hipHostRegisterDefault));

                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_C, (void *)C_data, 0));
                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_A, (void *)A_data, 0));
                gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_B, (void *)B_data, 0));
            }

            auto threads = gpu::block_size(elements), blocks = gpu::blocks(elements);

            detail::direct_product_kernel<DevDatatype<T>><<<threads, blocks, 0, gpu::get_stream()>>>(
                gpu_index_strides, gpu::HipCast<DevDatatype<T>, T>::cast(C_prefactor), gpu_C, gpu_C_strides,
                gpu::HipCast<DevDatatype<T>, T>::cast(AB_prefactor), gpu_A, gpu_A_strides, gpu_B, gpu_B_strides, elements, rank);

            gpu::stream_wait();

            gpu::hip_catch(hipFree((void *)gpu_index_strides));
            gpu::hip_catch(hipFree((void *)gpu_A_strides));
            gpu::hip_catch(hipFree((void *)gpu_B_strides));
            gpu::hip_catch(hipFree((void *)gpu_C_strides));

            if (this->_unit == einsums::python::detail::GPU_COPY) {
                gpu::hip_catch(hipFree((void *)gpu_A));
                gpu::hip_catch(hipFree((void *)gpu_B));
                gpu::hip_catch(hipMemcpy((void *)C_data, (const void *)gpu_C, C.shape(0) * C.strides(0), hipMemcpyDeviceToHost));

                gpu::hip_catch(hipFree((void *)gpu_C));
            } else {
                gpu::hip_catch(hipHostUnregister((void *)A_data));
                gpu::hip_catch(hipHostUnregister((void *)B_data));
                gpu::hip_catch(hipHostUnregister((void *)C_data));
            }
        } else {
#endif
            T       *C_data = (T *)C.mutable_unchecked<T>().mutable_data();
            const T *A_data = (const T *)A.unchecked<T>().data(), *B_data = (const T *)B.unchecked<T>().data();

#pragma omp parallel for simd
            for (size_t sentinel = 0; sentinel < elements; sentinel++) {
                size_t quotient = sentinel;
                size_t A_index = 0, B_index = 0, C_index = 0;
                for (int i = 0; i < rank; i++) {
                    size_t index = quotient / index_strides[i];
                    quotient %= index_strides[i];

                    A_index += (A.strides(i) / sizeof(T)) * index;
                    B_index += (B.strides(i) / sizeof(T)) * index;
                    C_index += (C.strides(i) / sizeof(T)) * index;
                }

                if (C_prefactor == T{0.0}) {
                    C_data[C_index] = AB_prefactor * A_data[A_index] * B_data[B_index];
                } else {
                    C_data[C_index] = C_prefactor * C_data[C_index] + AB_prefactor * A_data[A_index] * B_data[B_index];
                }
            }
#ifdef __HIP__
        }
#endif
    }

  public:
    PyEinsumDirectProductPlan() = delete;
    explicit PyEinsumDirectProductPlan(const PyEinsumGenericPlan &plan_base);
    PyEinsumDirectProductPlan(const PyEinsumDirectProductPlan &) = default;
    ~PyEinsumDirectProductPlan()                                 = default;

    void execute(const pybind11::object &C_prefactor, pybind11::array &C, const pybind11::object &AB_prefactor, const pybind11::array &A,
                 const pybind11::array &B) const override;
};

class PyEinsumGerPlan : public PyEinsumGenericPlan {
  private:
    bool _swap_AB;

    template <typename T>
    void execute_imp(T C_prefactor, pybind11::array &C, T AB_prefactor, const pybind11::array &A, const pybind11::array &B) const {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> && !std::is_same_v<T, std::complex<float>> &&
                      !std::is_same_v<T, std::complex<double>>) {
            this->execute_generic(C_prefactor, C, AB_prefactor, A, B);
        } else {

            const T *A_data = (const T *)A.unchecked<T>().data(), *B_data = (const T *)B.unchecked<T>().data();
            size_t   A_size = A.shape(0) * A.strides(0), B_size = B.shape(0) * B.strides(0), C_size = C.shape(0) * C.strides(0);

            size_t dC0 = 1, dC1 = 1;

            for (int i = 0; i < C.ndim(); i++) {
                for (int j = 0; j < A.ndim(); j++) {
                    if (this->_C_permute[i] == this->_A_permute[j]) {
                        dC0 *= A.shape(j);
                        break;
                    }
                }
            }

            for (int i = 0; i < C.ndim(); i++) {
                for (int j = 0; j < B.ndim(); j++) {
                    if (this->_C_permute[i] == this->_B_permute[j]) {
                        dC1 *= B.shape(j);
                        break;
                    }
                }
            }

            if (this->_swap_AB) {
                std::swap(dC0, dC1);
            }

            T *C_data = (T *)C.mutable_unchecked<T>().mutable_data();

#ifdef __HIP__
            if (this->_unit == einsums::python::detail::GPU_MAP || this->_unit == einsums::python::detail::GPU_COPY) {
                __device_ptr__ DevDatatype<T> *gpu_C, *gpu_A, *gpu_B, *gpu_AB_prefactor;

                if (this->_unit == einsums::python::detail::GPU_MAP) {
                    gpu::hip_catch(hipHostRegister((void *)A_data, A.nbytes(), hipHostRegisterDefault));
                    gpu::hip_catch(hipHostRegister((void *)B_data, B.nbytes(), hipHostRegisterDefault));
                    gpu::hip_catch(hipHostRegister((void *)C_data, C.nbytes(), hipHostRegisterDefault));

                    gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_C, (void *)C_data, 0));
                    gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_A, (void *)A_data, 0));
                    gpu::hip_catch(hipHostGetDevicePointer((void **)&gpu_B, (void *)B_data, 0));
                } else {
                    gpu::hip_catch(hipMalloc((void **)&gpu_A, A.nbytes()));
                    gpu::hip_catch(hipMalloc((void **)&gpu_B, B.nbytes()));
                    gpu::hip_catch(hipMalloc((void **)&gpu_C, C.nbytes()));

                    gpu::hip_catch(hipMemcpy((void *)gpu_A, (const void *)A_data, A.nbytes(), hipMemcpyHostToDevice));
                    gpu::hip_catch(hipMemcpy((void *)gpu_B, (const void *)B_data, B.nbytes(), hipMemcpyHostToDevice));
                    gpu::hip_catch(hipMemcpy((void *)gpu_C, (const void *)C_data, C.nbytes(), hipMemcpyHostToDevice));
                }

                gpu::hip_catch(hipMalloc((void **)&gpu_AB_prefactor, sizeof(T)));
                gpu::hip_catch(hipMemcpy((void *)gpu_AB_prefactor, &AB_prefactor, sizeof(T), hipMemcpyHostToDevice));

                // Scale the matrix.
                if (C_prefactor != T{1.0}) {
                    auto threads = gpu::block_size(C.size()), blocks = gpu::blocks(C.size());
                    detail::scale_array<<<threads, blocks, 0, gpu::get_stream()>>>(gpu::HipCast<DevDatatype<T>, T>::cast(C_prefactor),
                                                                                   gpu_C, C.size());
                }

                if (this->_swap_AB) {
                    std::swap(gpu_A, gpu_B);
                }

                if constexpr (std::is_same_v<T, float>) {
                    gpu::hipblas_catch(hipblasSger(gpu::get_blas_handle(), dC0, dC1, gpu_AB_prefactor, gpu_A, 1, gpu_B, 1, gpu_C, dC0));
                } else if constexpr (std::is_same_v<T, double>) {
                    gpu::hipblas_catch(hipblasDger(gpu::get_blas_handle(), dC0, dC1, gpu_AB_prefactor, gpu_A, 1, gpu_B, 1, gpu_C, dC0));
                } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                    gpu::hipblas_catch(hipblasCgeru(gpu::get_blas_handle(), dC0, dC1, (const hipblasComplex *)gpu_AB_prefactor,
                                                    (const hipblasComplex *)gpu_A, 1, (const hipblasComplex *)gpu_B, 1,
                                                    (hipblasComplex *)gpu_C, dC0));
                } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    gpu::hipblas_catch(hipblasZgeru(gpu::get_blas_handle(), dC0, dC1, (const hipblasDoubleComplex *)gpu_AB_prefactor,
                                                    (const hipblasDoubleComplex *)gpu_A, 1, (const hipblasDoubleComplex *)gpu_B, 1,
                                                    (hipblasDoubleComplex *)gpu_C, dC0));
                }

                gpu::stream_wait();

                if (this->_unit == einsums::python::detail::GPU_MAP) {
                    gpu::hip_catch(hipHostUnregister((void *)A_data));
                    gpu::hip_catch(hipHostUnregister((void *)B_data));
                    gpu::hip_catch(hipHostUnregister((void *)C_data));
                } else {
                    gpu::hip_catch(hipFree((void *)gpu_A));
                    gpu::hip_catch(hipFree((void *)gpu_B));

                    gpu::hip_catch(hipMemcpy((void *)C_data, (const void *)gpu_C, C.nbytes(), hipMemcpyDeviceToHost));
                    gpu::hip_catch(hipFree((void *)gpu_C));
                }

                gpu::hip_catch(hipFree((void *)gpu_AB_prefactor));
            } else {
#endif
                if (C_prefactor == T{0.0}) {
                    std::memset(C_data, 0, C.nbytes());
                } else if (C_prefactor != T{1.0}) {
                    EINSUMS_OMP_PARALLEL_FOR
                    for (size_t i = 0; i < C.size(); i++) {
                        C_data[i] *= C_prefactor;
                    }
                }

                if (this->_swap_AB) {
                    std::swap(A_data, B_data);
                }

                einsums::blas::ger(dC0, dC1, AB_prefactor, A_data, 1, B_data, 1, C_data, dC0);
#ifdef __HIP__
            }
#endif
        }
    }

  public:
    PyEinsumGerPlan() = delete;
    explicit PyEinsumGerPlan(bool swap_AB, const PyEinsumGenericPlan &plan_base);
    PyEinsumGerPlan(const PyEinsumGerPlan &) = default;
    ~PyEinsumGerPlan()                       = default;

    void execute(const pybind11::object &C_prefactor, pybind11::array &C, const pybind11::object &AB_prefactor, const pybind11::array &A,
                 const pybind11::array &B) const override;
};

class PyEinsumGemvPlan : public PyEinsumGenericPlan {
  public:
    PyEinsumGemvPlan() = delete;
    explicit PyEinsumGemvPlan(const PyEinsumGenericPlan &plan_base);
    PyEinsumGemvPlan(const PyEinsumGemvPlan &) = default;
    ~PyEinsumGemvPlan()                        = default;

    // void execute(float C_prefactor, pybind11::array_t<float> &C, float AB_prefactor, const pybind11::array_t<float> &A,
    //              const pybind11::array_t<float> &B) const override;

    // void execute(double C_prefactor, pybind11::array_t<double> &C, double AB_prefactor, const pybind11::array_t<double> &A,
    //              const pybind11::array_t<double> &B) const override;
};

class PyEinsumGemmPlan : public PyEinsumGenericPlan {
  public:
    PyEinsumGemmPlan() = delete;
    explicit PyEinsumGemmPlan(const PyEinsumGenericPlan &plan_base);
    PyEinsumGemmPlan(const PyEinsumGemmPlan &) = default;
    ~PyEinsumGemmPlan()                        = default;

    // void execute(float C_prefactor, pybind11::array_t<float> &C, float AB_prefactor, const pybind11::array_t<float> &A,
    //              const pybind11::array_t<float> &B) const override;

    // void execute(double C_prefactor, pybind11::array_t<double> &C, double AB_prefactor, const pybind11::array_t<double> &A,
    //              const pybind11::array_t<double> &B) const override;
};

EINSUMS_EXPORT std::shared_ptr<PyEinsumGenericPlan> compile_plan(std::string C_indices, std::string A_indices, std::string B_indices,
                                                                 python::detail::PyPlanUnit unit = python::detail::CPU);

} // namespace einsums::tensor_algebra

namespace einsums::python {
EINSUMS_EXPORT void export_tensor_algebra(pybind11::module_ &m);
}
