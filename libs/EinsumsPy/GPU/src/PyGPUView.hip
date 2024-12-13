#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <EinsumsPy/GPU/PyGPUView.hpp>
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <list>
#include <mutex>

using namespace einsums;
using namespace einsums::python;

struct MappingList;

static std::mutex   __constructor_lock;
static MappingList *__singleton = nullptr;

// Subject. Observed by PyGPUView.
struct MappingList final {
  private:
    struct MappingData {
      private:
        void                  *_pointer{nullptr};
        size_t                 _size{0};
        std::list<PyGPUView *> _observers;

      public:
        MappingData() = default;

        MappingData(void *pointer, size_t size) : _pointer{pointer}, _size{size} {
            hip_catch(hipHostRegister(pointer, size, hipHostRegisterDefault));
        }

        ~MappingData() {
            if (_pointer != nullptr) {
                hip_catch(hipHostUnregister(_pointer));
            }
        }

        void *get_pointer() const { return _pointer; }

        void *get_gpu_pointer(void const *host_pointer) const {
            uint8_t const *host_arit = (uint8_t const *)host_pointer, *dev_arit, *base_arit = (uint8_t const *)_pointer;

            hip_catch(hipHostGetDevicePointer((void **)&dev_arit, _pointer, 0));

            ptrdiff_t offset = host_arit - base_arit;

            return (void *)(dev_arit + offset);
        }

        size_t get_size() const { return _size; }

        bool has_refs() const { return _observers.size() > 0; }

        bool observed_by(PyGPUView &observer) const {
            for (auto item : _observers) {
                if (item == &observer) {
                    return true;
                }
            }

            return false;
        }

        void attach(PyGPUView &observer) { _observers.push_back(&observer); }

        void detach(PyGPUView &observer) { _observers.remove(&observer); }
    };
    std::list<MappingData> _pointers;
    std::recursive_mutex   _lock;

    MappingList()  = default;
    ~MappingList() = default;

    void refresh_list() {
        auto guard = std::lock_guard(_lock);
        _pointers.remove_if([](MappingData const &data) { return !data.has_refs(); });
    }

  public:
    static MappingList &get_singleton() {
        auto guard = std::lock_guard(__constructor_lock);
        if (__singleton == nullptr) {
            __singleton = new MappingList();
        }
        return *__singleton;
    }

    /**
     * Attach a PyGPUView to this subject with the given pointer and allocation size.
     *
     * @return The device pointer that maps the passed in pointer.
     */
    void *attach(PyGPUView &view, void *pointer, size_t size) {
        auto guard = std::lock_guard(_lock);
        bool found = false;
        for (auto &data : _pointers) {
            uint8_t const *base_arit = (uint8_t const *)data.get_pointer(), *ptr_arit = (uint8_t const *)pointer;
            ptrdiff_t      diff = ptr_arit - base_arit;

            if (diff >= 0 && diff < data.get_size() && diff + size <= data.get_size()) {
                data.attach(view);
                notify();
                return data.get_gpu_pointer(pointer);
            }
        }

        _pointers.emplace_back(pointer, size);
        _pointers.back().attach(view);

        notify();

        return _pointers.back().get_gpu_pointer(pointer);
    }

    /**
     * Remove a PyGPUView from this subject.
     */
    void detach(PyGPUView &view) {
        auto guard = std::lock_guard(_lock);
        for (auto &data : _pointers) {
            if (data.observed_by(view)) {
                data.detach(view);
            }
        }

        notify();
    }

    /**
     * Update the internal state. The name `notify` is the standard name for this behavior of a Subject from the Gang of Four,
     * even though this only actually updates items without any observers.
     */
    void notify() {
        auto guard = std::lock_guard(_lock);
        // No need to update dependents, since we only update the data without any observers.
        _pointers.remove_if([](MappingData const &data) { return !data.has_refs(); });
    }
};

PyGPUView::PyGPUView(pybind11::buffer &buffer, detail::PyViewMode mode) {
    auto buffer_info = buffer.request(true);

    _host_data = buffer_info.ptr;
    _rank      = buffer_info.ndim;
    _itemsize  = buffer_info.itemsize;
    _fmt_spec  = buffer_info.format;

    _mode = mode;

    if (_mode == detail::COPY) {
        hip_catch(hipMalloc((void **)&_dev_data, buffer_info.strides[0] * buffer_info.shape[0]));

        hip_catch(
            hipMemcpy((void *)_dev_data, (void const *)_host_data, buffer_info.strides[0] * buffer_info.shape[0], hipMemcpyHostToDevice));
    } else if (_mode == detail::MAP) {
        _dev_data = MappingList::get_singleton().attach(*this, _host_data, buffer_info.strides[0] * buffer_info.shape[0]);
    } else if (_mode == detail::DEVICE_TENSOR) {
        EINSUMS_THROW_EXCEPTION(
            enum_error, "Can not use DEVICE_TENSOR mode with buffer object. Only used to convert einsums::DeviceTensor to PyGPUView.");
    }

    _alloc_size = buffer_info.strides[0] * buffer_info.shape[0];
    _num_items  = 1;
    for (auto dim : buffer_info.shape) {
        _num_items *= dim;
    }

    _dims.resize(_rank);
    _strides.resize(_rank);

    for (int i = 0; i < _rank; i++) {
        _dims[i]    = buffer_info.shape[i];
        _strides[i] = buffer_info.strides[i];
    }

    hip_catch(hipMalloc((void **)&_gpu_dims, _rank * sizeof(size_t)));
    hip_catch(hipMalloc((void **)&_gpu_strides, _rank * sizeof(size_t)));
    hip_catch(hipMemcpy((void *)_gpu_dims, (void const *)_dims.data(), _rank * sizeof(size_t), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)_gpu_strides, (void const *)_strides.data(), _rank * sizeof(size_t), hipMemcpyHostToDevice));
}

PyGPUView::~PyGPUView() {
    if (_mode == detail::COPY) {
        // Don't copy before. There is no guarantee that the host data is still valid.
        hip_catch(hipFree(_dev_data));
    } else if (_mode == detail::MAP) {
        MappingList::get_singleton().detach(*this);
    } else if (_mode == detail::DEVICE_TENSOR) {
        hip_catch(hipFree(_gpu_strides));
    }

    if (_mode != detail::DEVICE_TENSOR) {
        hip_catch(hipFree(_gpu_dims));
        hip_catch(hipFree(_gpu_strides));
    }
}

void *PyGPUView::host_data() noexcept {
    return _host_data;
}

void const *PyGPUView::host_data() const noexcept {
    return _host_data;
}

void *PyGPUView::dev_data() noexcept {
    return _dev_data;
}

void const *PyGPUView::dev_data() const noexcept {
    return _dev_data;
}

std::vector<size_t> PyGPUView::dims() const noexcept {
    return _dims;
}

size_t PyGPUView::dim(int i) const {
    if (i < 0) {
        return _dims.at(i + _rank);
    }
    return _dims.at(i);
}

std::vector<size_t> PyGPUView::strides() const noexcept {
    return _strides;
}

size_t PyGPUView::stride(int i) const {
    if (i < 0) {
        return _strides.at(i + _rank);
    } else {
        return _strides.at(i);
    }
}

std::string PyGPUView::fmt_spec() const noexcept {
    return _fmt_spec;
}

size_t *PyGPUView::gpu_dims() noexcept {
    return _gpu_dims;
}

size_t const *PyGPUView::gpu_dims() const noexcept {
    return _gpu_dims;
}

size_t *PyGPUView::gpu_strides() noexcept {
    return _gpu_strides;
}

size_t const *PyGPUView::gpu_strides() const noexcept {
    return _gpu_strides;
}

size_t PyGPUView::rank() const noexcept {
    return _rank;
}

size_t PyGPUView::size() const noexcept {
    return _num_items;
}

size_t PyGPUView::itemsize() const noexcept {
    return _itemsize;
}

void PyGPUView::update_H2D() {
    if (_mode == detail::COPY) {
        hip_catch(hipMemcpy(_dev_data, _host_data, _alloc_size, hipMemcpyHostToDevice));
    }
}

void PyGPUView::update_D2H() {
    if (_mode == detail::COPY) {
        hip_catch(hipMemcpy(_host_data, _dev_data, _alloc_size, hipMemcpyDeviceToHost));
    }
}