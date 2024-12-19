#pragma once

#include <pybind11/pybind11.h>
// Pybind needs to come first.

#include <Einsums/Config.hpp>

#include <stdexcept>

namespace einsums::python::testing {

/**
 * @struct BadBuffer
 *
 * @brief A buffer with settable properties to allow for coverage of failure code.
 *
 * All of the fields of the buffer can be modified. The only one with restrictions is
 * the data pointer, which can only be set on initialization or cleared to NULL later.
 */
struct EINSUMS_EXPORT BadBuffer {
  private:
    void               *_ptr{nullptr};   /// Where the data is stored.
    size_t              _itemsize{0};    /// How big an item is.
    std::string         _format{""};     /// Format string.
    size_t              _ndim{0};        /// How many dimensions this buffer has.
    std::vector<size_t> _dims, _strides; /// The dimensions and strides. Strides are in bytes, dimensions are in elements.

    /**
     * The actual size of the buffer object. It can not be changed and is not dependent on the dimensions or strides.
     * This is because it is needed for copying the object itself in the copy constructor, as well as other management
     * duties.
     */
    size_t _size{0};

  public:
    BadBuffer() = default;
    BadBuffer(BadBuffer const &);

    /**
     * Construct a BadBuffer using the data from another Python buffer, such as a
     * NumPy array.
     */
    BadBuffer(pybind11::buffer const &buffer);

    ~BadBuffer();

    /*
     * Getters and setters. These should act normally.
     */

    /**
     * Get the pointer to the data.
     */
    void *get_ptr();

    /**
     * Get the pointer to the data, but make it const.
     */
    void const *get_ptr() const;

    /**
     * Set the pointer to the data to NULL. If there is anything stored in the old pointer, it will be freed.
     */
    void clear_ptr();

    /**
     * Get the number of bytes per item.
     */
    size_t get_itemsize() const;

    /**
     * Set the number of bytes per item.
     */
    void set_itemsize(size_t size);

    /**
     * Get the format string.
     */
    std::string get_format() const;

    /**
     * Set the format string.
     */
    void set_format(std::string const &str);

    /**
     * Get the number of dimensions.
     */
    size_t get_ndim() const;

    /**
     * Set the number of dimensions. This does not affect the shape or strides.
     */
    void set_ndim_noresize(size_t dim);

    /**
     * Set the number of dimensions and resize _dims and _strides to contain that many items.
     */
    void set_ndim(size_t dim);

    /**
     * Get the dimensions of the buffer.
     */
    std::vector<size_t> get_dims() const;

    /**
     * Set all of the dimensions of the buffer.
     */
    void set_dims(std::vector<size_t> const &dims);

    /**
     * Set the dimension of the buffer along a given axis.
     */
    void set_dim(int i, size_t dim);

    /**
     * Change the number of dimensions without affecting _ndim or _strides.
     */
    void change_dims_size(size_t new_size);

    /**
     * Get the strides in bytes of the buffer.
     */
    std::vector<size_t> get_strides() const;

    /**
     * Set all of the strides of the buffer.
     */
    void set_strides(std::vector<size_t> const &strides);

    /**
     * Set the stride of the buffer along the given axis.
     */
    void set_stride(int i, size_t stride);

    /**
     * Change the number of strides without affecting _ndim or _dims.
     */
    void change_strides_size(size_t new_size);
};

} // namespace einsums::python::testing