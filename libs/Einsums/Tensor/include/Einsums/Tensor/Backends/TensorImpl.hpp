//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "Einsums/Concepts/NamedRequirements.hpp"
#include "Einsums/Errors/Error.hpp"
#include "Einsums/Print.hpp"
#include "Einsums/TensorBase/Common.hpp"

namespace einsums {

namespace detail {

/**
 * @struct TensorData<T>
 *
 * @brief Underlying implementation details for tensors.
 *
 * @tparam T The data type being stored. It can be const or non-const. It can also be any numerical or complex type, though most
 * library functions only support float, double, std::complex<float>, and std::complex<double>.
 */
template <typename T>
struct TensorData final {
  public:
    using ValueType          = T;
    using ReferenceType      = T &;
    using ConstReferenceType = T const &;
    using PointerType        = T *;
    using ConstPointerType   = T const *;

    // Normal constructors. Note that the copy constructor only creates a copy of the view, not a new tensor with the same data.
    constexpr TensorData() noexcept = default;

    constexpr TensorData(TensorData<T> const &other)
        : data_{other.data_}, rank_{other.rank_}, size_{other.size_}, dims_{other.dims_}, strides_{other.strides_} {}

    constexpr TensorData(TensorData<T> &&other) noexcept
        : data_{other.data_}, rank_{other.rank_}, size_{other.size_}, dims_{std::move(other.dims_)}, strides_{std::move(other.strides_)} {
        other.data_ = nullptr;
        other.rank_ = 0;
        other.size_ = 0;
        other.dims_.clear();
        other.strides_.clear();
    }

    // Move and copy assignment. Note that the copy assignment only creates a copy of the view, not a new tensor with the same data.
    constexpr TensorData<T> &operator=(TensorData<T> const &other) {
        data_ = other.data_;
        rank_ = other.rank_;
        size_ = other.size_;
        dims_.resize(rank_);
        strides_.resize(rank_);

        dims_.assign(other.dims_.cbegin(), other.dims_.cend());
        strides_.assign(other.strides_.cbegin(), other.strides_.cend());
    }

    constexpr TensorData<T> &operator=(TensorData<T> &&other) {
        data_    = other.data_;
        rank_    = other.rank_;
        size_    = other.size_;
        dims_    = std::move(other.dims_);
        strides_ = std::move(other.strides_);

        other.data_ = nullptr;
        other.rank_ = 0;
        other.size_ = 0;
        other.dims_.clear();
        other.strides_.clear();
    }

    // Destructor.
    constexpr ~TensorData() noexcept {
        data_ = nullptr;
        rank_ = 0;
        size_ = 0;
        dims_.clear();
        strides_.clear();
    }

    // Now the more useful constructors.
    template <ContainerOrInitializer Dims>
    constexpr TensorData(T *data, Dims const &dims, bool row_major = false)
        : data_{data}, dims_(dims.begin(), dims.end()), strides_(dims.size()), rank_{dims.size()} {

        size_ = dims_to_strides(dims_, strides_, row_major);
    }

    template <ContainerOrInitializer Dims, ContainerOrInitializer Strides>
    constexpr TensorData(T *data, Dims const &dims, Strides const &strides)
        : data_{data}, dims_(dims.cbegin(), dims.cend()), strides_(strides.begin(), strides.end()), rank_{dims.size()}, size_{1} {
        for (int i = 0; i < rank_; i++) {
            size_ *= dims_[i];
        }
    }

    // Getters

    constexpr size_t rank() const noexcept { return rank_; }

    constexpr size_t size() const noexcept { return size_; }

    constexpr std::vector<size_t> const &dims() const noexcept { return dims_; }

    constexpr std::vector<size_t> const &strides() const noexcept { return strides_; }

    constexpr T *data() noexcept { return data_; }

    constexpr T const *data() const noexcept { return data_; }

    // Indexed getters.

    constexpr size_t dim(int i) const {
        int temp = i;
        if (temp < 0) {
            temp += rank_;
        }

        if (temp < 0 || temp >= rank_) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to dim is out of range! Expected between {} and {}, got {}.",
                                    -rank_, rank_ - 1, i);
        }

        return dims_[temp];
    }

    constexpr size_t stride(int i) const {
        int temp = i;
        if (temp < 0) {
            temp += rank_;
        }

        if (temp < 0 || temp >= rank_) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to stride is out of range! Expected between {} and {}, got {}.",
                                    -rank_, rank_ - 1, i);
        }

        return strides_[temp];
    }

    // Indexed data retrieval.
    template <std::integral... MultiIndex>
    constexpr T *data_no_check(MultiIndex &&...index) {
        return data_no_check(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T *data_no_check(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>([&offset, &index, this](size_t n) { offset += std::get<n>(index) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T *data_no_check(MultiIndex const &index) {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        return data_ + std::inner_product(index.begin(), index.end(), strides_.cbegin(), 0);
    }

    template <std::integral... MultiIndex>
    constexpr T const *data_no_check(MultiIndex &&...index) const {
        return data_no_check(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T const *data_no_check(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>([&offset, &index, this](size_t n) { offset += std::get<n>(index) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T const *data_no_check(MultiIndex const &index) const {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        return data_ + std::inner_product(index.begin(), index.end(), strides_.cbegin(), 0);
    }

    template <std::integral... MultiIndex>
    constexpr T *data(MultiIndex &&...index) {
        return data(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T *data(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>(
            [&offset, &index, this](size_t n) { offset += adjust_index(std::get<n>(index), dims_[n], n) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T *data(MultiIndex const &index) {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        size_t offset = 0;

        for (int i = 0; i < index.size(); i++) {
            offset += adjust_index(index[i], dims_[i], i) * strides_[i];
        }

        return data_ + offset;
    }

    template <std::integral... MultiIndex>
    constexpr T const *data(MultiIndex &&...index) const {
        return data(std::make_tuple(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    constexpr T const *data(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        size_t offset = 0;
        for_sequence<sizeof...(MultiIndex)>(
            [&offset, &index, this](size_t n) { offset += adjust_index(std::get<n>(index), dims_[n], n) * strides_[n]; });

        return data_ + offset;
    }

    template <ContainerOrInitializer MultiIndex>
    constexpr T const *data(MultiIndex const &index) const {
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices in the container passed to data!");
        }

        size_t offset = 0;

        for (int i = 0; i < index.size(); i++) {
            offset += adjust_index(index[i], dims_[i], i) * strides_[i];
        }

        return data_ + offset;
    }

    // Const conversion.
    constexpr operator TensorData<T const>() { return TensorData<T const>(data_, dims_, strides_); }

    // Subscripting.
    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript_no_check(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript_no_check(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ReferenceType subscript_no_check(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript_no_check(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript_no_check(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ConstReferenceType subscript_no_check(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data_no_check(index);
    }

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ReferenceType subscript(std::tuple<MultiIndex...> const &index) {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ReferenceType subscript(Index const &index) {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(std::forward<MultiIndex>(index)...);
    }

    template <std::integral... MultiIndex>
    constexpr ConstReferenceType subscript(std::tuple<MultiIndex...> const &index) const {
        if (sizeof...(MultiIndex) < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (sizeof...(MultiIndex) > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    template <ContainerOrInitializer Index>
    constexpr ConstReferenceType subscript(Index const &index) const {
        if (index.size() < rank_) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to subscript tensor!");
        }
        if (index.size() > rank_) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript tensor!");
        }
        return *data(index);
    }

    // View creation.
    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> subscript_no_check(MultiIndex &&...index) {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> subscript_no_check(std::tuple<MultiIndex...> const &index) {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, index);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> const subscript_no_check(MultiIndex &&...index) const {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> const subscript_no_check(std::tuple<MultiIndex...> const &index) const {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<false, 0>(out_dims, out_strides, index);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> subscript(MultiIndex &&...index) {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> subscript(std::tuple<MultiIndex...> const &index) {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, index);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> const subscript(MultiIndex &&...index) const {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, std::forward<MultiIndex>(index)...);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

    template <typename... MultiIndex>
        requires(!std::is_integral_v<MultiIndex> || ... || false)
    constexpr TensorData<T> const subscript(std::tuple<MultiIndex...> const &index) const {
        std::vector<size_t> out_dims{}, out_strides{};

        out_dims.reserve(rank_);
        out_strides.reserve(rank_);

        size_t offset = compute_view<true, 0>(out_dims, out_strides, index);

        return TensorData<T>(data_ + offset, out_dims, out_strides);
    }

#ifdef EINSUMS_COMPUTE_CODE

    void copy_to_gpu() {
        std::vector<size_t> index_strides(rank_);

        dims_to_strides(dims_, index_strides);

        if (size_ < 2048) {
            std::vector<T> buffer(size_);

            for (size_t i = 0; i < size_; i++) {
                size_t sentinel;

                sentinel_to_sentinels(i, index_strides, strides_, sentinel);

                buffer[i] = data_[sentinel];
            }

            hip_catch(hipMemcpy((void *)gpu_ptr_, (void const *)buffer.data(), size_ * sizeof(T), hipMemcpyHostToDevice));
        } else {
            // Double buffer bigger transactions.
            std::vector<T, BufferAllocator<T>> buffer1(1024), buffer2(1024);
            std::binary_semaphore              buffer1_semaphore(1), buffer2_semaphore(1);
            bool                               buffer1_ready = false, buffer2_ready = false;

#    pragma omp parallel
            {
#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (buffer1_ready) {
                            std::yield();
                        }
                        buffer1_semaphore.acquire();
                        for (size_t j = 0; j < std::min(size_ - i, 1024); j++) {
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
                            std::yield();
                        }

                        buffer2_semaphore.acquire();

                        for (size_t j = 0; j < std::min(size_ - i - 1024, 1024); j++) {
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
                    std::yield();
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (!buffer1_ready) {
                            std::yield();
                        }
                        buffer1_semaphore.acquire();

                        hip_catch(
                            hipMemcpy((void *)(gpu_ptr_ + i), (void *)buffer1.data(), buffer1.size() * sizeof(T), hipMemcpyHostToDevice));

                        buffer1_ready = false;

                        buffer1_semaphore.release();

                        while (!buffer2_ready) {
                            std::yield();
                        }

                        buffer2_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)(gpu_ptr_ + i + 1024), (void *)buffer2.data(), buffer2.size() * sizeof(T),
                                            hipMemcpyHostToDevice));

                        buffer2_ready = false;

                        buffer2_semaphore.release();
                    }
                }
#    pragma omp taskgroup
            }
        }
    }

    void copy_from_gpu() {

        std::vector<size_t> index_strides(rank_);

        dims_to_strides(dims_, index_strides);

        if (size_ < 2048) {
            std::vector<T> buffer(size_);

            hip_catch(hipMemcpy((void *)buffer.data(), (void const *)gpu_ptr_, size_ * sizeof(T), hipMemcpyDeviceToHost));

            for (size_t i = 0; i < size_; i++) {
                size_t sentinel;

                sentinel_to_sentinels(i, index_strides, strides_, sentinel);

                buffer[i]       = data_[sentinel];
                data_[sentinel] = buffer[i];
            }

        } else {
            // Double buffer bigger transactions.
            std::vector<T, BufferAllocator<T>> buffer1(1024), buffer2(1024);
            std::binary_semaphore              buffer1_semaphore(1), buffer2_semaphore(1);
            bool                               buffer1_ready = true, buffer2_ready = true;

#    pragma omp parallel
            {
#    pragma omp task
                {
                    for (size_t i = 0; i < size_; i += 2048) {
                        while (buffer1_ready) {
                            std::yield();
                        }
                        buffer1_semaphore.acquire();
                        for (size_t j = 0; j < std::min(size_ - i, 1024); j++) {
                            size_t sentinel;
                            sentinel_to_sentinels(i + j, index_strides, strides_, sentinel);
                            data_[sentinel] = buffer1[j];
                        }

                        buffer1_ready = true;

                        buffer1_semaphore.release();

                        while (buffer2_ready) {
                            std::yield();
                        }

                        buffer2_semaphore.acquire();

                        for (size_t j = 0; j < std::min(size_ - i - 1024, 1024); j++) {
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
                            std::yield();
                        }
                        buffer1_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)buffer1.data(), (void const *)(gpu_ptr_ + i), buffer1.size() * sizeof(T),
                                            hipMemcpyDeviceToHost));

                        buffer1_ready = false;

                        buffer1_semaphore.release();

                        while (!buffer2_ready) {
                            std::yield();
                        }

                        buffer2_semaphore.acquire();

                        hip_catch(hipMemcpy((void *)buffer2.data(), (void const *)(gpu_ptr_ + i + 1024), buffer2.size() * sizeof(T),
                                            hipMemcpyDeviceToHost));

                        buffer2_ready = false;

                        buffer2_semaphore.release();
                    }
                }
#    pragma omp taskgroup
            }
        }
    }

    gpu::GPUPointer<T> request_gpu_ptr(bool copy_on_create = true) {
        if (gpu_ptr_) {
            return gpu_ptr_;
        } else {
            gpu_ptr_ = allocator_.allocate(size_);

            if (copy_on_create) {
                copy_to_gpu();
            }

            return gpu_ptr_;
        }
    }

    void free_gpu_ptr(bool copy_on_destroy = true) {
        if (gpu_ptr_) {
            if (copy_on_destroy) {
                copy_from_gpu();
            }

            allocator_.deallocate(gpu_ptr_, size_);
            gpu_ptr_ = nullptr;
        }
    }
#endif

  private:
    template <bool CheckInds, size_t I>
    constexpr size_t compute_view(std::vector<size_t> &out_dims, std::vector<size_t> &out_strides) {
        return 0;
    }

    template <bool CheckInds, size_t I, std::integral First, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t> &out_dims, std::vector<size_t> &out_strides, First first, Rest &&...rest) {
        auto index = first;
        if constexpr (CheckInds) {
            index = adjust_index(index, dims_[I], I);
        }

        return index * strides_[I] + compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t> &out_dims, std::vector<size_t> &out_strides, Range const &first, Rest &&...rest) {
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

    template <bool CheckInds, size_t I, typename... Rest>
    constexpr size_t compute_view(std::vector<size_t> &out_dims, std::vector<size_t> &out_strides, AllT const &, Rest &&...rest) {

        out_dims.push_back(dims_[I]);
        out_strides.push_back(strides_[I]);

        return compute_view<CheckInds, I + 1>(out_dims, out_strides, std::forward<Rest>(rest)...);
    }

    template <bool CheckInds, size_t I, typename... MultiIndex>
    constexpr size_t compute_view(std::vector<size_t> &out_dims, std::vector<size_t> &out_strides,
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

    std::vector<size_t> dims_{}, strides_{};

#ifdef EINSUMS_COMPUTE_CODE
    GPUAllocator<T> allocator_{};
    GPUPointer<T>   gpu_ptr_{nullptr};
#endif
};

} // namespace detail

} // namespace einsums