//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUMemory/GPUAllocator.hpp>
#    include <Einsums/GPUMemory/GPUMemoryTracker.hpp>
#    include <Einsums/GPUMemory/GPUPointer.hpp>
#    include <Einsums/GPUStreams/GPUStreams.hpp>
#    include <Einsums/hipBLAS.hpp>

#    include <hip/hip_common.h>
#    include <hip/hip_complex.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {

/**
 * @brief Adjust the index if it is negative and raise an error if it is out of range.
 *
 * The @c index_position parameter is used for creating the exception message. If it is negative,
 * then the exception message will look something like <tt>"The index is out of range! Expected between -5 and 4, got 6!"</tt>.
 * However, for better diagnostics, a non-negative index_position, for example 2, will give an exception message like
 * <tt>"The third index is out of range! Expected between -5 and 4, got 6!"</tt>. Note that these are zero-indexed, so
 * passing in 2 prints out "third".
 *
 * @param index The index to adjust and check.
 * @param dim The dimension to compare to.
 * @param index_position Used for the error message. If it is negative, then the index position
 * will not be included in the error message.
 */
template <std::integral IntType>
constexpr size_t adjust_index(IntType index, size_t dim, int index_position = -1) {
    if constexpr (std::is_signed_v<IntType>) {
        auto hold = index;

        if (hold < 0) {
            hold += dim;
        }

        if (hold < 0 || hold >= dim) {
            if (index_position < 0) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index is out of range! Expected between {} and {}, got {}!",
                                        -(ptrdiff_t)dim, dim - 1, index);
            } else {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index is out of range! Expected between {} and {}, got {}!",
                                        print::ordinal<int>(index_position + 1), -(ptrdiff_t)dim, dim - 1, index);
            }
        }
        return hold;
    } else {
        if (index >= dim) {
            if (index_position < 0) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index is out of range! Expected between {} and {}, got {}!", 0, dim - 1,
                                        index);
            } else {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "The {} index is out of range! Expected between {} and {}, got {}!",
                                        print::ordinal<int>(index_position + 1), 0, dim - 1, index);
            }
        }

        return index;
    }
}

template <typename T>
struct TensorImpl;

namespace detail {

/**
 * @struct TensorImpl<T>
 *
 * @brief Underlying implementation details for tensors.
 *
 * @tparam T The data type being stored. It can be const or non-const. It can also be any numerical or complex type, though most
 * library functions only support float, double, std::complex<float>, and std::complex<double>.
 */
template <typename T>
struct TensorImpl final {
  public:
    using ValueType          = T;
    using ReferenceType      = T &;
    using ConstReferenceType = T const &;
    using PointerType        = T *;
    using ConstPointerType   = T const *;

    // Normal constructors. Note that the copy constructor only creates a copy of the view, not a new tensor with the same data.

    /**
     * @brief Default constructor.
     */
    constexpr TensorImpl() noexcept = default;

    /**
     * @brief Copy constructor.
     *
     * This creates a new view of the other tensor implementation. This means that the pointer the new object will
     * contain is the same as the pointer the other object contains.
     */
    constexpr TensorImpl(TensorImpl<T> const &other);

    /**
     * @brief Move constructor.
     *
     * At exit, the input tensor will no longer have valid data. Its pointer will be the null pointer and its other data will be cleared.
     */
    constexpr TensorImpl(TensorImpl<T> &&other) noexcept;

    // Move and copy assignment. Note that the copy assignment only creates a copy of the view, not a new tensor with the same data.

    /**
     * @brief Copy assignment.
     *
     * This creates a new view of the other tensor implementation. This means that the pointer this object will contain
     * is the same as the pointer that the other object conatins.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> const &other);

    /**
     * @brief Move assignment.
     *
     * At exit, the input tensor will no longer have valid data. Its pointer will be the null pointer and its other data will be cleared.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> &&other);

    // Destructor.
    constexpr ~TensorImpl() noexcept;

    // Now the more useful constructors.

    /**
     * @brief Create a new tensor implementation that wraps the given pointer and has the given dimensions.
     *
     * A stride ordering can be chosen. By default, it is column major.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the tensor.
     * @param row_major If true, calculate the strides in row-major order. Otherwise, calculate the strides in column-major order.
     */
    template <ContainerOrInitializer Dims>
    constexpr TensorImpl(T *data, Dims const &dims, bool row_major = false)
        : data_{data}, dims_(dims.begin(), dims.end()), strides_(dims.size()), rank_{dims.size()} {

        size_          = dims_to_strides(dims_, strides_, row_major);
        is_contiguous_ = true;
    }

    /**
     * @brief Create a new tensor implementation that wraps the given pointer and has the given dimensions and strides.
     *
     * @param data The pointer to wrap.
     * @param dims The dimensions of the tensor.
     * @param strides The strides of the tensor in number of elements.
     */
    template <ContainerOrInitializer Dims, ContainerOrInitializer Strides>
    constexpr TensorImpl(T *data, Dims const &dims, Strides const &strides);

    // Getters

    /**
     * @brief Get the rank of the tensor.
     */
    constexpr size_t rank() const noexcept { return rank_; }

    /**
     * @brief Get the size of the tensor in number of elements.
     */
    constexpr size_t size() const noexcept { return size_; }

    /**
     * @brief Get the dimensions of the tensor.
     */
    constexpr BufferVector<size_t> const &dims() const noexcept { return dims_; }

    /**
     * @brief Get the strides of the tensor.
     */
    constexpr BufferVector<size_t> const &strides() const noexcept { return strides_; }

    /**
     * @brief Get the data pointer.
     */
    constexpr T *data() noexcept { return data_; }

    /**
     * @brief Get the data pointer.
     */
    constexpr T const *data() const noexcept { return data_; }

    /**
     * @brief Check if the tensor is contiguous.
     */
    constexpr bool is_contiguous() const noexcept { return is_contiguous_; }

    // Setters

    /**
     * @brief Set the data to a new pointer.
     */
    constexpr void reset_data(T *data) noexcept { data_ = data; }

    /**
     * @brief Set the data to a new pointer.
     */
    constexpr void reset_data(T const *data) noexcept { data_ = data; }

    // Indexed getters.

    /**
     * @brief Get the dimension along a certain axis.
     *
     * This method handles negative indexing. If the index is negative, then it will be handled as
     * indexing from the end of the tensor rather than the beginning.
     *
     * @param i The axis to query.
     */
    constexpr size_t dim(int i) const;

    /**
     * @brief Get the stride along a certain axis.
     *
     * This method handles negative indexing. If the index is negative, then it will be handled as
     * indexing from the end of the tensor rather than the beginning.
     *
     * @param i The axis to query.
     */
    constexpr size_t stride(int i) const;

    // Indexed data retrieval.
    template <std::integral... MultiIndex>
    constexpr T *data_no_check(MultiIndex &&...index);

    template <std::integral... MultiIndex>
    constexpr T *data_no_check(std::tuple<MultiIndex...> const &index);

    template <ContainerOrInitializer MultiIndex>
    constexpr T *data_no_check(MultiIndex const &index);

    template <std::integral... MultiIndex>
    constexpr T const *data_no_check(MultiIndex &&...index) const;

    template <std::integral... MultiIndex>
    constexpr T const *data_no_check(std::tuple<MultiIndex...> const &index) const;

    template <ContainerOrInitializer MultiIndex>
    constexpr T const *data_no_check(MultiIndex const &index) const;

    template <std::integral... MultiIndex>
    constexpr T *data(MultiIndex &&...index);

    template <std::integral... MultiIndex>
    constexpr T *data(std::tuple<MultiIndex...> const &index);

    template <ContainerOrInitializer MultiIndex>
    constexpr T *data(MultiIndex const &index);

    template <std::integral... MultiIndex>
    constexpr T const *data(MultiIndex &&...index) const;

    template <std::integral... MultiIndex>
    constexpr T const *data(std::tuple<MultiIndex...> const &index) const;

    template <ContainerOrInitializer MultiIndex>
    constexpr T const *data(MultiIndex const &index) const;

    // Const conversion.
    constexpr operator TensorImpl<T const>() { return TensorImpl<T const>(data_, dims_, strides_); }

    // Subscripting.
    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript_no_check(MultiIndex &&...index);

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript_no_check(std::tuple<MultiIndex...> const &index);

    template <ContainerOrInitializer Index>
    constexpr ReferenceType subscript_no_check(Index const &index);

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript_no_check(MultiIndex &&...index) const;

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript_no_check(std::tuple<MultiIndex...> const &index) const;

    template <ContainerOrInitializer Index>
    constexpr ConstReferenceType subscript_no_check(Index const &index) const;

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript(MultiIndex &&...index);

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript(std::tuple<MultiIndex...> const &index);

    template <ContainerOrInitializer Index>
    constexpr ReferenceType subscript(Index const &index);

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript(MultiIndex &&...index) const;

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript(std::tuple<MultiIndex...> const &index) const;

    template <ContainerOrInitializer Index>
    constexpr ConstReferenceType subscript(Index const &index) const;

    // View creation.
    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript_no_check(MultiIndex &&...index);

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript_no_check(std::tuple<MultiIndex...> const &index);

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript_no_check(MultiIndex &&...index) const;

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript_no_check(std::tuple<MultiIndex...> const &index) const;

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript(MultiIndex &&...index);

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> subscript(std::tuple<MultiIndex...> const &index);

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript(MultiIndex &&...index) const;

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorImpl<T> const subscript(std::tuple<MultiIndex...> const &index) const;

    template <typename TOther>
    void copy_from_both_contiguous(TensorImpl<TOther> const &other) ;

    template <typename TOther>
    void copy_from(TensorImpl<TOther> const &other);

    template <typename TOther>
    void add_assign_both_contiguous(TensorImpl<TOther> const &other);

    template <typename TOther>
    void add_assign(TensorImpl<TOther> const &other);

    template <typename TOther>
    void sub_assign_both_contiguous(TensorImpl<TOther> const &other);

    template <typename TOther>
    void sub_assign(TensorImpl<TOther> const &other);

    template <typename TOther>
    void mul_assign_both_contiguous(TensorImpl<TOther> const &other);

    template <typename TOther>
    void mul_assign(TensorImpl<TOther> const &other);
    template <typename TOther>
    void div_assign_both_contiguous(TensorImpl<TOther> const &other);

    template <typename TOther>
    void div_assign(TensorImpl<TOther> const &other);

    template <typename TOther>
    void add_assign_scalar_contiguous(TOther value);

    template <typename TOther>
    void add_assign_scalar(TOther value);

    template <typename TOther>
    void sub_assign_scalar_contiguous(TOther value);

    template <typename TOther>
    void sub_assign_scalar(TOther value);

    template <typename TOther>
    void mul_assign_scalar_contiguous(TOther value);

    template <typename TOther>
    void mul_assign_scalar(TOther value);

    template <typename TOther>
    void div_assign_scalar_contiguous(TOther value);

    template <typename TOther>
    void div_assign_scalar(TOther value);

#ifdef EINSUMS_COMPUTE_CODE

    void copy_to_gpu(gpu::GPUPointer<T> gpu_ptr) {
        BufferVector<size_t> index_strides(rank_);

        dims_to_strides(dims_, index_strides);

        if (size_ < 2048) {
            BufferVector<T> buffer(size_);

            for (size_t i = 0; i < size_; i++) {
                size_t sentinel;

                sentinel_to_sentinels(i, index_strides, strides_, sentinel);

                buffer[i] = data_[sentinel];
            }

            hip_catch(hipMemcpy((void *)gpu_ptr, (void const *)buffer.data(), size_ * sizeof(T), hipMemcpyHostToDevice));
        } else {
            // Double buffer bigger transactions.
            BufferVector<T>       buffer1(1024), buffer2(1024);
            std::binary_semaphore buffer1_semaphore(1), buffer2_semaphore(1);
            bool                  buffer1_ready = false, buffer2_ready = false;

#    pragma omp parallel
            {
#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();
                        for (size_t j = 0; j < std::min(size_ - i, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j, index_strides, strides_, sentinel);
                            buffer1[j] = data_[sentinel];
                        }

                        if (size_ - i < 1024) {
                            buffer1.resize(size_ - i);
                        }

                        buffer1_ready = true;

                        buffer1_semaphore.release();

                        while (buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        for (size_t j = 0; j < std::min(size_ - i - 1024, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j + 1024, index_strides, strides_, sentinel);
                            buffer2[j] = data_[sentinel];
                        }

                        if (size_ - i - 1024 < 1024) {
                            buffer2.resize(size_ - i);
                        }

                        buffer2_ready = true;

                        buffer2_semaphore.release();
                    }
                }

#    pragma omp task
                {
                    std::this_thread::yield();
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (!buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();

                        hip_catch(
                            hipMemcpy((void *)(gpu_ptr + i), (void *)buffer1.data(), buffer1.size() * sizeof(T), hipMemcpyHostToDevice));

                        buffer1_ready = false;

                        buffer1_semaphore.release();

                        while (!buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)(gpu_ptr + i + 1024), (void *)buffer2.data(), buffer2.size() * sizeof(T),
                                            hipMemcpyHostToDevice));

                        buffer2_ready = false;

                        buffer2_semaphore.release();
                    }
                }
#    pragma omp taskgroup
            }
        }
    }

    void copy_from_gpu(gpu::GPUPointer<T> gpu_ptr) {

        BufferVector<size_t> index_strides(rank_);

        dims_to_strides(dims_, index_strides);

        if (size_ < 2048) {
            BufferVector<T> buffer(size_);

            hip_catch(hipMemcpy((void *)buffer.data(), (void const *)gpu_ptr, size_ * sizeof(T), hipMemcpyDeviceToHost));

            for (size_t i = 0; i < size_; i++) {
                size_t sentinel;

                sentinel_to_sentinels(i, index_strides, strides_, sentinel);

                buffer[i]       = data_[sentinel];
                data_[sentinel] = buffer[i];
            }

        } else {
            // Double buffer bigger transactions.
            BufferVector<T>       buffer1(1024), buffer2(1024);
            std::binary_semaphore buffer1_semaphore(1), buffer2_semaphore(1);
            bool                  buffer1_ready = true, buffer2_ready = true;

#    pragma omp parallel
            {
#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();
                        for (size_t j = 0; j < std::min(size_ - i, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j, index_strides, strides_, sentinel);
                            data_[sentinel] = buffer1[j];
                        }

                        buffer1_ready = true;

                        buffer1_semaphore.release();

                        while (buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        for (size_t j = 0; j < std::min(size_ - i - 1024, (size_t)1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j + 1024, index_strides, strides_, sentinel);
                            buffer2[j]      = data_[sentinel];
                            data_[sentinel] = buffer2[j];
                        }

                        buffer2_ready = true;

                        buffer2_semaphore.release();
                    }
                }

#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (!buffer1_ready) {
                            std::this_thread::yield();
                        }
                        buffer1_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)buffer1.data(), (void const *)(gpu_ptr + i), buffer1.size() * sizeof(T),
                                            hipMemcpyDeviceToHost));

                        buffer1_ready = false;

                        buffer1_semaphore.release();

                        while (!buffer2_ready) {
                            std::this_thread::yield();
                        }

                        buffer2_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)buffer2.data(), (void const *)(gpu_ptr + i + 1024), buffer2.size() * sizeof(T),
                                            hipMemcpyDeviceToHost));

                        buffer2_ready = false;

                        buffer2_semaphore.release();
                    }
                }
#    pragma omp taskgroup
            }
        }
    }

    gpu::GPUPointer<T> request_gpu_ptr() {
        auto [ptr, do_copy] = gpu::GPUMemoryTracker::get_singleton().get_pointer<T>(gpu_handle_, size());

        if (do_copy) {
            copy_to_gpu(ptr);
        }

        return ptr;
    }

    void release_gpu_ptr(gpu::GPUPointer<T> &ptr) {
        if (ptr) {
            gpu::GPUMemoryTracker::get_singleton().release_handle(gpu_handle_);
        }
    }
#endif

  private:
    void gpu_init() {
#ifdef EINSUMS_COMPUTE_CODE
        gpu_handle_ = gpu::GPUMemoryTracker::get_singleton().create_handle();
#endif
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides) {
        return 0;
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, std::integral First, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides, First first,
                                  Rest &&...rest) {
        auto index = first;
        if constexpr (CheckInds) {
            index = adjust_index(index, dims_[I], I);
        }

        return index * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides, Range const &first,
                                  Rest &&...rest) {
        auto index = first;
        if constexpr (CheckInds) {
            index[0] = adjust_index(index[0], dims_[I], I);
            index[1] = adjust_index(index[1], dims_[I] + 1, I);
        }

        if (index[0] > index[1]) {
            auto temp = index[0];
            index[0]  = index[1];
            index[1]  = temp;
        }

        out_dims.push_back(index[1] - index[0]);
        out_strides.push_back(strides_[I]);

        return index[0] * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides, AllT const &,
                                  Rest &&...rest) {

        out_dims.push_back(dims_[I]);
        out_strides.push_back(strides_[I]);

        return compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename Alloc1, typename Alloc2, typename... MultiIndex>
    constexpr size_t compute_view(std::vector<size_t, Alloc1> &out_dims, std::vector<size_t, Alloc2> &out_strides,
                                  std::tuple<MultiIndex...> const &indices) {
        using CurrType = typename std::tuple_element_t<I, std::tuple<MultiIndex...>>;
        size_t index   = 0;
        if constexpr (I >= sizeof...(MultiIndex)) {
            return 0;
        } else {
            if constexpr (std::is_integral_v<CurrType>) {
                index = std::get<I>(indices);
                if constexpr (CheckInds) {
                    index = adjust_index(index, dims_[I], I);
                }

            } else if constexpr (std::is_same_v<Range, CurrType>) {
                auto range = std::get<I>(indices);
                if constexpr (CheckInds) {
                    range[0] = adjust_index(range[0], dims_[I], I);
                    range[1] = adjust_index(range[1], dims_[I] + 1, I);
                }

                if (range[0] > range[1]) {
                    auto temp = range[0];
                    range[0]  = range[1];
                    range[1]  = temp;
                }

                index = range[0];
                out_dims.push_back(range[1] - range[0]);
                out_strides.push_back(strides_[I]);
            } else if constexpr (std::is_same_v<AllT, CurrType>) {
                out_dims.push_back(dims_[I]);
                out_strides.push_back(strides_[I]);
            }
            return index * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, indices);
        }
    }

    T *data_{nullptr};

    size_t rank_{0}, size_{0};

    BufferVector<size_t> dims_{}, strides_{};

#ifdef EINSUMS_COMPUTE_CODE
    size_t gpu_handle_{0};
#endif

    bool is_contiguous_{false};
};

} // namespace detail

} // namespace einsums