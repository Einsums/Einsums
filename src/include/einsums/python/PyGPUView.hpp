#pragma once

#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include "einsums/_GPUUtils.hpp"

#include "einsums/DeviceTensor.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <vector>

namespace einsums::python {

namespace detail {

/**
 * @enum PyViewMode
 *
 * @brief Details how to handle the data viewed by the PyGPUView object.
 */
enum PyViewMode {
    /**
     * @brief Map the data into the GPU address space.
     * With this mode, data synchronization is handled by the memory bus.
     * However, since each time the cache is invalidated a memory transfer happens,
     * operations with this can end up being rather slow.
     */
    MAP,
    /**
     * @brief Copy the data into GPU memory.
     * With this, memory is not automatically synchronized. If you need to synchronize
     * the memory, use the methods PyGPUView::update_H2D and PyGPUView::update_D2H.
     * This method may be faster than mapping directly, but it is limited by the size of
     * the GPU memory.
     */
    COPY,
    /**
     * @brief Indicates that the PyGPUView object is viewing an einsums::DeviceTensor<T, Rank> object.
     */
    DEVICE_TENSOR
};

} // namespace detail

class PyGPUView;

using SharedPyGPUView = std::shared_ptr<PyGPUView>;

/**
 * @class PyGPUView
 *
 * @brief Makes the data in a Python buffer object available to the GPU.
 */
class PyGPUView {
  private:
    void *_host_data;
    void *_dev_data;

    std::vector<size_t> _dims, _strides;
    size_t             *_gpu_dims, *_gpu_strides;
    size_t              _rank;
    size_t              _itemsize;
    size_t              _alloc_size, _num_items;

    std::string _fmt_spec;

    detail::PyViewMode                _mode;
    einsums::detail::HostToDeviceMode _dev_tensor_mode{einsums::detail::UNKNOWN};

  public:
    /**
     * @brief Creates a view of the given buffer object.
     *
     * This makes the data in the buffer available to the GPU with the given mode.
     *
     * @param buffer The buffer to make available.
     * @param mode The mode for making the data available. Can not be detail::DEVICE_TENSOR.
     */
    PyGPUView(pybind11::buffer &buffer, detail::PyViewMode mode = detail::COPY)
        THROWS(einsums::EinsumsException, einsums::gpu::detail::ErrorOutOfMemory, einsums::gpu::detail::ErrorInvalidValue,
               einsums::gpu::detail::ErrorUnknown, std::bad_alloc, pybind11::error_already_set);

    /**
     * @brief Creates a view of the given tensor.
     *
     * This makes the data in the DeviceTensor available to be used alongside Python buffers.
     *
     * @param tensor The tensor to convert.
     */
    template <typename T, size_t Rank>
    PyGPUView(DeviceTensor<T, Rank> &tensor) THROWS(einsums::gpu::detail::ErrorOutOfMemory, einsums::gpu::detail::ErrorInvalidValue,
                                                    einsums::gpu::detail::ErrorUnknown, std::bad_alloc)
        : _dims{tensor.dims()}, _strides(Rank), _gpu_dims{tensor.gpu_dims()}, _rank{Rank}, _itemsize{sizeof(T)},
          _alloc_size{sizeof(T) * tensor.dims(0) * tensor.strides(0)}, _num_items{tensor.size()}, _mode{detail::DEVICE_TENSOR},
          _fmt_spec{pybind11::format_descriptor<T>::format()}, _host_data{(void *)tensor.host_data()}, _dev_data{(void *)tensor.data()},
          _dev_tensor_mode{tensor.mode()} {
        gpu::hip_catch(hipMalloc((void **)&_gpu_strides, Rank * sizeof(size_t)));

        for (int i = 0; i < Rank; i++) {
            _strides[i] = sizeof(T) * tensor.strides(i);
        }

        gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), sizeof(T) * Rank, hipMemcpyHostToDevice));
    }

    /**
     * @brief Creates a view of the given tensor.
     *
     * This makes the data in the DeviceTensor available to be used alongside Python buffers.
     *
     * @param tensor The tensor to convert.
     */
    template <typename T, size_t Rank>
    PyGPUView(DeviceTensorView<T, Rank> &tensor) THROWS(einsums::gpu::detail::ErrorOutOfMemory, einsums::gpu::detail::ErrorInvalidValue,
                                                        einsums::gpu::detail::ErrorUnknown, std::bad_alloc)
        : _dims{tensor.dims()}, _strides(Rank), _gpu_dims{tensor.gpu_dims()}, _rank{Rank}, _itemsize{sizeof(T)},
          _alloc_size{sizeof(T) * tensor.dims(0) * tensor.strides(0)}, _num_items{tensor.size()}, _mode{detail::DEVICE_TENSOR},
          _fmt_spec{pybind11::format_descriptor<T>::format()}, _host_data{(void *)tensor.host_data()}, _dev_data{(void *)tensor.data()},
          _dev_tensor_mode{tensor.mode()} {
        gpu::hip_catch(hipMalloc((void **)&_gpu_strides, Rank * sizeof(size_t)));

        for (int i = 0; i < Rank; i++) {
            _strides[i] = sizeof(T) * tensor.strides(i);
        }

        gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), sizeof(T) * Rank, hipMemcpyHostToDevice));
    }

    /**
     * @brief Creates a view of the given tensor.
     *
     * This makes the data in the DeviceTensor available to be used alongside Python buffers.
     *
     * @param tensor The tensor to convert.
     */
    template <typename T, size_t Rank>
    PyGPUView(const DeviceTensor<T, Rank> &tensor) THROWS(einsums::gpu::detail::ErrorOutOfMemory, einsums::gpu::detail::ErrorInvalidValue,
                                                          einsums::gpu::detail::ErrorUnknown, std::bad_alloc)
        : _dims{tensor.dims()}, _strides(Rank), _gpu_dims{tensor.gpu_dims()}, _rank{Rank}, _itemsize{sizeof(T)},
          _alloc_size{sizeof(T) * tensor.dims(0) * tensor.strides(0)}, _num_items{tensor.size()}, _mode{detail::DEVICE_TENSOR},
          _fmt_spec{pybind11::format_descriptor<T>::format()} {
        gpu::hip_catch(hipMalloc((void **)&_gpu_strides, Rank * sizeof(size_t)));

        for (int i = 0; i < Rank; i++) {
            _strides[i] = sizeof(T) * tensor.strides(i);
        }

        gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), sizeof(T) * Rank, hipMemcpyHostToDevice));
    }

    /**
     * @brief Creates a view of the given tensor.
     *
     * This makes the data in the DeviceTensor available to be used alongside Python buffers.
     *
     * @param tensor The tensor to convert.
     */
    template <typename T, size_t Rank>
    PyGPUView(const DeviceTensorView<T, Rank> &tensor)
        THROWS(einsums::gpu::detail::ErrorOutOfMemory, einsums::gpu::detail::ErrorInvalidValue, einsums::gpu::detail::ErrorUnknown,
               std::bad_alloc)
        : _dims{tensor.dims()}, _strides(Rank), _gpu_dims{tensor.gpu_dims()}, _rank{Rank}, _itemsize{sizeof(T)},
          _alloc_size{sizeof(T) * tensor.dims(0) * tensor.strides(0)}, _num_items{tensor.size()}, _mode{detail::DEVICE_TENSOR},
          _fmt_spec{pybind11::format_descriptor<T>::format()} {
        gpu::hip_catch(hipMalloc((void **)&_gpu_strides, Rank * sizeof(size_t)));

        for (int i = 0; i < Rank; i++) {
            _strides[i] = sizeof(T) * tensor.strides(i);
        }

        gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), sizeof(T) * Rank, hipMemcpyHostToDevice));
    }

    PyGPUView(const PyGPUView &) = delete;

    ~PyGPUView() THROWS(einsums::gpu::detail::ErrorInvalidDevicePointer, einsums::gpu::detail::ErrorInvalidValue,
                        einsums::gpu::detail::ErrorHostMemoryNotRegistered);

    /**
     * @brief Get the pointer to the host data.
     */
    void *host_data() noexcept;

    /**
     * @brief Get the pointer to the host data.
     */
    const void *host_data() const noexcept;

    /**
     * @brief Get the pointer to the device data.
     */
    void *dev_data() noexcept;

    /**
     * @brief Get the pointer to the device data.
     */
    const void *dev_data() const noexcept;

    /**
     * @brief Get the dimensions of each axis.
     */
    std::vector<size_t> dims() const noexcept;

    /**
     * @brief Get the dimension of the given axis. Supports negative indexing.
     */
    size_t dim(int i) const THROWS(std::out_of_range);

    /**
     * @brief Get the strides of each axis in bytes.
     *
     * This is different from how the C++ side works, which is in elements. To get
     * the equivalent C++ stride, you must divide by PyGPUView::itemsize().
     */
    std::vector<size_t> strides() const noexcept;

    /**
     * @brief Get the strides of the given axis in bytes. Supports negative indexing.
     *
     * This is different from how the C++ side works, which is in elements. To get
     * the equivalent C++ stride, you must divide by PyGPUView::itemsize().
     */
    size_t stride(int i) const THROWS(std::out_of_range);

    /**
     * @brief Get the format specifier for the data type.
     *
     * This is specified by pybind11. For instance, floats are "f", doubles are "d", and long doubles are "g".
     * Complex types are "Z" followed by their underlying type.
     */
    std::string fmt_spec() const noexcept;

    /**
     * @brief Get the device pointer to the array of the dimensions.
     */
    size_t *gpu_dims() noexcept;

    /**
     * @brief Get the device pointer to the array of the dimensions.
     */
    const size_t *gpu_dims() const noexcept;

    /**
     * @brief Get the device pointer to the array of the strides in bytes.
     */
    size_t *gpu_strides() noexcept;

    /**
     * @brief Get the device pointer to the array of the strides in bytes.
     */
    const size_t *gpu_strides() const noexcept;

    /**
     * @brief Get the rank.
     */
    size_t rank() const noexcept;

    /**
     * @brief Get the number of elements.
     */
    size_t size() const noexcept;

    /**
     * @brief Get the size of each item.
     */
    size_t itemsize() const noexcept;

    /**
     * @brief Synchronize the data by sending the host data to the device.
     *
     * This only does something when the mode is set to detail::COPY. Otherwise, it does nothing.
     */
    void update_H2D() THROWS(einsums::gpu::detail::ErrorInvalidValue, einsums::gpu::detail::ErrorUnknown);

    /**
     * @brief Synchronize the data by sending the device data to the host.
     *
     * This only does something when the mode is set to detail::COPY. Otherwise, it does nothing.
     */
    void update_D2H() THROWS(einsums::gpu::detail::ErrorInvalidValue, einsums::gpu::detail::ErrorUnknown);
};

/**
 * @brief Make the symbols in this file available to Python.
 */
EINSUMS_EXPORT void export_gpu_view(pybind11::module_ &mod);

} // namespace einsums::python
