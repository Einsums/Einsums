//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <omp.h>

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

#include <algorithm>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::element_operations)

BEGIN_EINSUMS_NAMESPACE_HPP(detail)
template <typename vector, typename Functor>
void omp_loop(vector &data, Functor functor) {
    LabeledSection0();

    // TODO: This only works for Tensors not their views because we assume data is a std::vector
    EINSUMS_OMP_PARALLEL {
        auto tid       = omp_get_thread_num();
        auto chunksize = data.size() / omp_get_num_threads();
        auto begin     = data.begin() + chunksize * tid;
        auto end       = (tid == omp_get_num_threads() - 1) ? data.end() : begin + chunksize;

        EINSUMS_OMP_SIMD
        for (auto i = begin; i < end; i++) {
            *i = functor(*i);
        }
    }
}

#ifdef __HIP__

template <typename T, size_t Rank>
__global__ void sum_kernel(T *out, const T *data, size_t *dims, size_t *strides, size_t size);

template <typename T, size_t Rank>
__global__ void max_kernel(T *out, const T *data, size_t *dims, size_t *strides, size_t size);

template <typename T, size_t Rank>
__global__ void min_kernel(T *out, const T *data, size_t *dims, size_t *strides, size_t size);

template <typename T, size_t Rank>
__global__ void abs_kernel(T *data, size_t *dims, size_t *strides, size_t size);

template <typename T, size_t Rank>
__global__ void invert_kernel(T *data, size_t *dims, size_t *strides, size_t size);

template <typename T, size_t Rank>
__global__ void exp_kernel(T *data, size_t *dims, size_t *strides, size_t size);

template <typename T, size_t Rank>
__global__ void scale_kernel(T *data, T scale, size_t *dims, size_t *strides, size_t size);

#endif
END_EINSUMS_NAMESPACE_HPP(detail)

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto sum(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        T result{0};
#pragma omp parallel for reduction(+ : result)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            if (tensor.block_dim(i) == 0) {
                continue;
            }
            result += sum(tensor[i]);
        }
        return result;
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t workers = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z * sizeof(T);

        T *gpu_result;

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&gpu_result, sizeof(T), stream));
        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        detail::sum_kernel<T, Rank><<<blocks, threads, workers, stream>>>(gpu_result, tensor.data(), dims_and_strides, dims_and_strides + Rank,
                                                                 tensor.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        T result;
        gpu::hip_catch(hipMemcpy((void *)&result, gpu_result, sizeof(T), hipMemcpyDeviceToHost));

        gpu::hip_catch(hipFreeAsync(gpu_result, stream));

        return result;

#endif
    } else {
        T result{0};

#pragma omp parallel for reduction(+ : result)
        for (ssize_t i = 0; i < tensor.size(); i++) {
            size_t index = 0, quotient = i;

#pragma unroll
            for (int j = 0; j < Rank; j++) {
                index += tensor.stride(j) * (quotient % tensor.dim(j));
                quotient /= tensor.dim(j);
            }

            result += tensor.data()[index];
        }

        return result;
    }
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto max(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        std::vector<T> max_arr(tensor.num_blocks());

#pragma omp parallel for shared(max_arr)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            max_arr[i] = (tensor.block_dim(i) == 0) ? T{-INFINITY} : max(tensor[i]);
        }
        return *std::max_element(max_arr.begin(), max_arr.end());
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t workers = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z * sizeof(T);

        T *gpu_result;

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&gpu_result, sizeof(T), stream));
        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        detail::max_kernel<T, Rank><<<blocks, threads, workers, stream>>>(gpu_result, tensor.data(), dims_and_strides, dims_and_strides + Rank,
                                                                 tensor.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        T result;
        gpu::hip_catch(hipMemcpy((void *)&result, gpu_result, sizeof(T), hipMemcpyDeviceToHost));

        gpu::hip_catch(hipFreeAsync(gpu_result, stream));

        return result;

#endif
    } else {
        std::vector<T> max_arr(omp_get_max_threads());

        for (int i = 0; i < max_arr.size(); i++) {
            max_arr[i] = T{-INFINITY};
        }

#pragma omp parallel shared(max_arr)
        {
            auto tid = omp_get_thread_num();
            for (int i = omp_get_thread_num(); i < tensor.size(); i += omp_get_num_threads()) {
                size_t index = 0, quotient = i;

#pragma unroll
                for (int j = 0; j < Rank; j++) {
                    index += tensor.stride(j) * (quotient % tensor.dim(j));
                    quotient /= tensor.dim(j);
                }

                max_arr[tid] = (max_arr[tid] > tensor.data()[index]) ? max_arr[tid] : tensor.data()[index];
            }
        }

        return *std::max_element(max_arr.begin(), max_arr.end());
    }
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto min(const TensorType<T, Rank> &tensor) -> T {
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        std::vector<T> min_arr(tensor.num_blocks());

#pragma omp parallel for shared(min_arr)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            min_arr[i] = (tensor.block_dim(i) == 0) ? T{INFINITY} : min(tensor[i]);
        }
        return *std::min_element(min_arr.begin(), min_arr.end());
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t workers = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z * sizeof(T);

        T *gpu_result;

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&gpu_result, sizeof(T), stream));
        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        detail::min_kernel<T, Rank><<<blocks, threads, workers, stream>>>(gpu_result, tensor.data(), dims_and_strides, dims_and_strides + Rank,
                                                                 tensor.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        T result;
        gpu::hip_catch(hipMemcpy((void *)&result, gpu_result, sizeof(T), hipMemcpyDeviceToHost));

        gpu::hip_catch(hipFreeAsync(gpu_result, stream));

        return result;

#endif
    } else {
        std::vector<T> min_arr(omp_get_max_threads());

        for (int i = 0; i < min_arr.size(); i++) {
            min_arr[i] = T{INFINITY};
        }

#pragma omp parallel shared(min_arr)
        {
            auto tid = omp_get_thread_num();
            for (int i = omp_get_thread_num(); i < tensor.size(); i += omp_get_num_threads()) {
                size_t index = 0, quotient = i;

#pragma unroll
                for (int j = 0; j < Rank; j++) {
                    index += tensor.stride(j) * (quotient % tensor.dim(j));
                    quotient /= tensor.dim(j);
                }

                min_arr[tid] = (min_arr[tid] < tensor.data()[index]) ? min_arr[tid] : tensor.data()[index];
            }
        }

        return *std::min_element(min_arr.begin(), min_arr.end());
    }
}

BEGIN_EINSUMS_NAMESPACE_HPP(new_tensor)

using einsums::element_operations::max; // nolint
using einsums::element_operations::sum; // nolint

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto abs(const TensorType<T, Rank> &tensor)
    -> std::conditional_t<
        einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>,
#ifdef __HIP__
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, BlockTensor<T, Rank>,
                           BlockDeviceTensor<T, Rank>>,
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, Tensor<T, Rank>, DeviceTensor<T, Rank>>> {
#else
        BlockTensor<T, Rank>, Tensor<T, Rank>> {
#endif
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);
#pragma omp parallel for shared(result)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            result[i] = abs(tensor[i]);
        }
        return result;
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);

        // Copy
        result = tensor;

        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        einsums::element_operations::detail::abs_kernel<T, Rank><<<blocks, threads, 0, stream>>>(result.data(), dims_and_strides,
                                                                                        dims_and_strides + Rank, result.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        return result;

#endif
    } else {
        auto result = create_tensor_like(tensor);

#pragma omp parallel shared(result)
        {
            for (int i = omp_get_thread_num(); i < tensor.size(); i += omp_get_num_threads()) {
                size_t from_index = 0, to_index = 0, quotient = i;

#pragma unroll
                for (int j = 0; j < Rank; j++) {
                    from_index += tensor.stride(j) * (quotient % tensor.dim(j));
                    to_index += result.stride(j) * (quotient % tensor.dim(j));
                    quotient /= tensor.dim(j);
                }

                result.data()[to_index] = std::abs(tensor.data()[from_index]);
            }
        }

        return result;
    }
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto invert(const TensorType<T, Rank> &tensor)
    -> std::conditional_t<
        einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>,
#ifdef __HIP__
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, BlockTensor<T, Rank>,
                           BlockDeviceTensor<T, Rank>>,
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, Tensor<T, Rank>, DeviceTensor<T, Rank>>> {
#else
        BlockTensor<T, Rank>, Tensor<T, Rank>> {
#endif
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);
#pragma omp parallel for shared(result)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            result[i] = invert(tensor[i]);
        }
        return result;
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);

        // Copy
        result = tensor;

        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        einsums::element_operations::detail::invert_kernel<T, Rank><<<blocks, threads, 0, stream>>>(result.data(), dims_and_strides,
                                                                                           dims_and_strides + Rank, result.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        return result;

#endif
    } else {
        auto result = create_tensor_like(tensor);

#pragma omp parallel shared(result)
        {
            for (int i = omp_get_thread_num(); i < tensor.size(); i += omp_get_num_threads()) {
                size_t from_index = 0, to_index = 0, quotient = i;

#pragma unroll
                for (int j = 0; j < Rank; j++) {
                    from_index += tensor.stride(j) * (quotient % tensor.dim(j));
                    to_index += result.stride(j) * (quotient % tensor.dim(j));
                    quotient /= tensor.dim(j);
                }

                result.data()[to_index] = T{1} / tensor.data()[from_index];
            }
        }

        return result;
    }
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto exp(const TensorType<T, Rank> &tensor)
    -> std::conditional_t<
        einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>,
#ifdef __HIP__
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, BlockTensor<T, Rank>,
                           BlockDeviceTensor<T, Rank>>,
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, Tensor<T, Rank>, DeviceTensor<T, Rank>>> {
#else
        BlockTensor<T, Rank>, Tensor<T, Rank>> {
#endif
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);
#pragma omp parallel for shared(result)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            result[i] = exp(tensor[i]);
        }
        return result;
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);

        // Copy
        result = tensor;

        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        einsums::element_operations::detail::exp_kernel<T, Rank><<<blocks, threads, 0, stream>>>(result.data(), dims_and_strides,
                                                                                        dims_and_strides + Rank, result.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        return result;

#endif
    } else {
        auto result = create_tensor_like(tensor);

#pragma omp parallel shared(result)
        {
            for (int i = omp_get_thread_num(); i < tensor.size(); i += omp_get_num_threads()) {
                size_t from_index = 0, to_index = 0, quotient = i;

#pragma unroll
                for (int j = 0; j < Rank; j++) {
                    from_index += tensor.stride(j) * (quotient % tensor.dim(j));
                    to_index += result.stride(j) * (quotient % tensor.dim(j));
                    quotient /= tensor.dim(j);
                }

                result.data()[to_index] = std::exp(tensor.data()[from_index]);
            }
        }

        return result;
    }
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto scale(const T &scale, const TensorType<T, Rank> &tensor)
    -> std::conditional_t<
        einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>,
#ifdef __HIP__
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, BlockTensor<T, Rank>,
                           BlockDeviceTensor<T, Rank>>,
        std::conditional_t<einsums::detail::IsIncoreRankTensorV<TensorType<T, Rank>, Rank, T>, Tensor<T, Rank>, DeviceTensor<T, Rank>>> {
#else
        BlockTensor<T, Rank>, Tensor<T, Rank>> {
#endif
    LabeledSection0();

    if constexpr (einsums::detail::IsBlockRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);
#pragma omp parallel for shared(result)
        for (int i = 0; i < tensor.num_blocks(); i++) {
            result[i] = einsums::element_operations::new_tensor::scale(scale, tensor[i]);
        }
        return result;
#ifdef __HIP__
    } else if constexpr (einsums::detail::IsDeviceRankTensorV<TensorType<T, Rank>, Rank, T>) {
        auto result = create_tensor_like(tensor);

        // Copy
        result = tensor;

        auto blocks = gpu::blocks(tensor.size()), threads = gpu::block_size(tensor.size());

        size_t *dims_and_strides;

        hipStream_t stream = gpu::get_stream();

        gpu::hip_catch(hipMallocAsync((void **)&dims_and_strides, 2 * Rank * sizeof(size_t), stream));

        gpu::hip_catch(hipMemcpyAsync(dims_and_strides, tensor.dims().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));
        gpu::hip_catch(
            hipMemcpyAsync(dims_and_strides + Rank, tensor.strides().data(), Rank * sizeof(size_t), hipMemcpyHostToDevice, stream));

        einsums::element_operations::detail::scale_kernel<T, Rank><<<blocks, threads, 0, stream>>>(result.data(), scale, dims_and_strides,
                                                                                          dims_and_strides + Rank, result.size());

        gpu::hip_catch(hipFreeAsync(dims_and_strides, stream));

        gpu::stream_wait();

        return result;

#endif
    } else {
        auto result = create_tensor_like(tensor);

#pragma omp parallel shared(result)
        {
            for (int i = omp_get_thread_num(); i < tensor.size(); i += omp_get_num_threads()) {
                size_t from_index = 0, to_index = 0, quotient = i;

#pragma unroll
                for (int j = 0; j < Rank; j++) {
                    from_index += tensor.stride(j) * (quotient % tensor.dim(j));
                    to_index += result.stride(j) * (quotient % tensor.dim(j));
                    quotient /= tensor.dim(j);
                }

                result.data()[to_index] = scale * tensor.data()[from_index];
            }
        }

        return result;
    }
}

END_EINSUMS_NAMESPACE_HPP(new_tensor)
END_EINSUMS_NAMESPACE_HPP(einsums::element_operations)

#ifdef __HIP__
#    include "einsums/gpu/ElementKernels.hpp"
#endif