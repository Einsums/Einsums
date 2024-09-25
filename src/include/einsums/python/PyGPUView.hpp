#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <pybind11/pybind11.h>
#include <source_location>

namespace einsums::python {

namespace detail {

enum PyViewMode { MAP, COPY, DEVICE_TENSOR };

} // namespace detail

class PyGPUView;

using SharedPyGPUView = std::shared_ptr<PyGPUView>;

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
    PyGPUView(pybind11::buffer &buffer, detail::PyViewMode mode = detail::COPY);

    PyGPUView(const pybind11::buffer &buffer, detail::PyViewMode mode = detail::COPY);

    template <typename T, size_t Rank>
    PyGPUView(DeviceTensor<T, Rank> &tensor)
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

    template <typename T, size_t Rank>
    PyGPUView(DeviceTensorView<T, Rank> &tensor)
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

    template <typename T, size_t Rank>
    PyGPUView(const DeviceTensor<T, Rank> &tensor)
        : _dims{tensor.dims()}, _strides(Rank), _gpu_dims{tensor.gpu_dims()}, _rank{Rank}, _itemsize{sizeof(T)},
          _alloc_size{sizeof(T) * tensor.dims(0) * tensor.strides(0)}, _num_items{tensor.size()}, _mode{detail::DEVICE_TENSOR},
          _fmt_spec{pybind11::format_descriptor<T>::format()} {
        gpu::hip_catch(hipMalloc((void **)&_gpu_strides, Rank * sizeof(size_t)));

        for (int i = 0; i < Rank; i++) {
            _strides[i] = sizeof(T) * tensor.strides(i);
        }

        gpu::hip_catch(hipMemcpy((void *)_gpu_strides, (const void *)_strides.data(), sizeof(T) * Rank, hipMemcpyHostToDevice));
    }

    template <typename T, size_t Rank>
    PyGPUView(const DeviceTensorView<T, Rank> &tensor)
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

    ~PyGPUView();

    void *host_data();

    const void *host_data() const;

    void *dev_data();

    const void *dev_data() const;

    std::vector<size_t> dims() const;

    size_t dim(int i) const;

    std::vector<size_t> strides() const;

    size_t stride(int i) const;

    std::string fmt_spec() const;

    size_t *gpu_dims();

    const size_t *gpu_dims() const;

    size_t *gpu_strides();

    const size_t *gpu_strides() const;

    size_t rank() const;

    size_t size() const;

    size_t itemsize() const;

    void update_H2D();
    void update_D2H();
};

EINSUMS_EXPORT void export_gpu_view(pybind11::module_ &mod);

} // namespace einsums::python
