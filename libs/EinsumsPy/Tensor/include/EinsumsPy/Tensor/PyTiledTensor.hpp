//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTiledTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#include <memory>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <source_location>
#include <stdexcept>

namespace einsums::python {

template <typename T>
void export_tiled_tensor(pybind11::module_ &mod) {

    namespace py = pybind11;

    std::string suffix = "";

    if constexpr (std::is_same_v<T, float>) {
        suffix = "F";
    } else if constexpr (std::is_same_v<T, double>) {
        suffix = "D";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        suffix = "C";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        suffix = "Z";
    }

    auto PyTiledTensor =
        py::class_<RuntimeTiledTensor<T>, std::shared_ptr<RuntimeTiledTensor<T>>>(mod, ("TiledTensor" + suffix).c_str())
            .def(py::init<RuntimeTiledTensor<T> const &>())
            .def(py::init<std::string const &, size_t, std::vector<size_t> const &>())
            .def(py::init<std::string const &, std::vector<std::vector<size_t>> const &>())
            .def(
                "tile",
                [](RuntimeTiledTensor<T> &self, py::args args) -> RuntimeTensor<T> & {
                    if (args.size() != self.rank()) {
                        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                "Wrong number of arguments passed to tiled tensor to get tile! Expected {}, got {}.",
                                                self.rank(), args.size());
                    }

                    std::vector<ptrdiff_t> cast_args(self.rank());

                    for (size_t i = 0; i < self.rank(); i++) {
                        cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                    }

                    return self.tile(cast_args);
                },
                py::return_value_policy::reference_internal)
            .def(
                "tile",
                [](RuntimeTiledTensor<T> &self, std::vector<ptrdiff_t> const &index) -> RuntimeTensor<T> & {
                    if (index.size() != self.rank()) {
                        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                "Wrong number of indices passed to tiled tensor to get tile! Expected {}, got {}.",
                                                self.rank(), index.size());
                    }

                    return self.tile(index);
                },
                py::return_value_policy::reference_internal)
            .def("set_tile",
                 [](RuntimeTiledTensor<T> &self, std::vector<ptrdiff_t> const &index, RuntimeTensor<T> const &copy) {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to get tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }

                     self.tile(index) = copy;
                 })
            .def("set_tile",
                 [](RuntimeTiledTensor<T> &self, std::vector<ptrdiff_t> const &index, RuntimeTensorView<T> const &copy) {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to get tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }

                     self.tile(index) = copy;
                 })
            .def("has_tile",
                 [](RuntimeTiledTensor<T> &self, py::args args) -> bool {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled tensor to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.has_tile(cast_args);
                 })
            .def("has_tile",
                 [](RuntimeTiledTensor<T> &self, std::vector<ptrdiff_t> const &index) -> bool {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to check tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }
                     return self.has_tile(index);
                 })
            .def("tile_of",
                 [](RuntimeTiledTensor<T> &self, py::args args) -> std::vector<size_t> {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled tensor to get tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.tile_of(cast_args);
                 })
            .def("tile_of",
                 [](RuntimeTiledTensor<T> &self, std::vector<ptrdiff_t> const &index) -> std::vector<size_t> {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to get tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }

                     return self.tile_of(index);
                 })
            .def("has_zero_size",
                 [](RuntimeTiledTensor<T> &self, py::args args) -> bool {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled tensor to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.has_zero_size(cast_args);
                 })
            .def("has_zero_size",
                 [](RuntimeTiledTensor<T> &self, std::vector<ptrdiff_t> const &index) -> bool {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to check tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }
                     return self.has_zero_size(index);
                 })
            .def("__getitem__",
                 [](RuntimeTiledTensor<T> const &self, py::tuple args) -> T {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled tensor to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.at(cast_args);
                 })
            .def("__setitem__",
                 [](RuntimeTiledTensor<T> &self, py::tuple args, T value) {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled tensor to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     self(cast_args) = value;
                 })
            .def("zero", &RuntimeTiledTensor<T>::zero)
            .def("zero_no_clear", &RuntimeTiledTensor<T>::zero_no_clear)
            .def("set_all", &RuntimeTiledTensor<T>::set_all)
            .def("set_all_existing", &RuntimeTiledTensor<T>::set_all_existing)
            .def(py::self += T())
            .def(py::self -= T())
            .def(py::self *= T())
            .def(py::self /= T())
            .def(py::self += RuntimeTiledTensor<T>())
            .def(py::self -= RuntimeTiledTensor<T>())
            .def(py::self *= RuntimeTiledTensor<T>())
            .def(py::self /= RuntimeTiledTensor<T>())
            .def_property_readonly("tile_offsets", &RuntimeTiledTensor<T>::tile_offsets)
            .def_property_readonly("tile_sizes", &RuntimeTiledTensor<T>::tile_sizes)
            .def_property("name", &RuntimeTiledTensor<T>::name, &RuntimeTiledTensor<T>::set_name)
            .def("size", &RuntimeTiledTensor<T>::size)
            .def("grid_size", static_cast<size_t (RuntimeTiledTensor<T>::*)() const>(&RuntimeTiledTensor<T>::grid_size))
            .def("grid_size", static_cast<size_t (RuntimeTiledTensor<T>::*)(int) const>(&RuntimeTiledTensor<T>::grid_size))
            .def("num_filled", &RuntimeTiledTensor<T>::num_filled)
            .def("__len__", &RuntimeTiledTensor<T>::num_filled)
            .def("full_view_of_underlying", &RuntimeTiledTensor<T>::full_view_of_underlying)
            .def_property_readonly("shape", &RuntimeTiledTensor<T>::dims)
            .def("__iter__", [](RuntimeTiledTensor<T> &self) { return py::make_iterator(self.tiles().begin(), self.tiles().end()); })
            .def("copy", [](RuntimeTiledTensor<T> const &self) -> RuntimeTiledTensor<T> { return self; })
            .def("__copy__", [](RuntimeTiledTensor<T> const &self) -> RuntimeTiledTensor<T> { return self; })
            .def("deepcopy", [](RuntimeTiledTensor<T> const &self) -> RuntimeTiledTensor<T> { return self; })
            .def("__deepcopy__", [](RuntimeTiledTensor<T> const &self) -> RuntimeTiledTensor<T> { return self; })
            .def("rank", &RuntimeTiledTensor<T>::rank);

    auto PyTiledTensorView =
        py::class_<RuntimeTiledTensorView<T>, std::shared_ptr<RuntimeTiledTensorView<T>>>(mod, ("TiledTensorView" + suffix).c_str())
            .def(py::init<RuntimeTiledTensorView<T> const &>())
            .def(py::init<std::string const &, size_t, std::vector<size_t> const &>())
            .def(py::init<std::string const &, std::vector<std::vector<size_t>>>())
            .def(
                "tile",
                [](RuntimeTiledTensorView<T> &self, py::args args) -> RuntimeTensorView<T> & {
                    if (args.size() != self.rank()) {
                        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                "Wrong number of arguments passed to tiled TensorView to get tile! Expected {}, got {}.",
                                                self.rank(), args.size());
                    }

                    std::vector<ptrdiff_t> cast_args(self.rank());

                    for (size_t i = 0; i < self.rank(); i++) {
                        cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                    }

                    return self.tile(cast_args);
                },
                py::return_value_policy::reference_internal)
            .def(
                "tile",
                [](RuntimeTiledTensorView<T> &self, std::vector<ptrdiff_t> const &index) -> RuntimeTensorView<T> & {
                    if (index.size() != self.rank()) {
                        EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                "Wrong number of indices passed to tiled TensorView to get tile! Expected {}, got {}.",
                                                self.rank(), index.size());
                    }

                    return self.tile(index);
                },
                py::return_value_policy::reference_internal)
            .def("set_tile",
                 [](RuntimeTiledTensorView<T> &self, std::vector<ptrdiff_t> const &index, RuntimeTensor<T> const &copy) {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to get tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }

                     self.tile(index) = copy;
                 })
            .def("set_tile",
                 [](RuntimeTiledTensorView<T> &self, std::vector<ptrdiff_t> const &index, RuntimeTensorView<T> const &copy) {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled tensor to get tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }

                     self.tile(index) = copy;
                 })
            .def("has_tile",
                 [](RuntimeTiledTensorView<T> &self, py::args args) -> bool {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled TensorView to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.has_tile(cast_args);
                 })
            .def("has_tile",
                 [](RuntimeTiledTensorView<T> &self, std::vector<ptrdiff_t> const &index) -> bool {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled TensorView to check tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }
                     return self.has_tile(index);
                 })
            .def("tile_of",
                 [](RuntimeTiledTensorView<T> &self, py::args args) -> std::vector<size_t> {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled TensorView to get tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.tile_of(cast_args);
                 })
            .def("tile_of",
                 [](RuntimeTiledTensorView<T> &self, std::vector<ptrdiff_t> const &index) -> std::vector<size_t> {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled TensorView to get tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }

                     return self.tile_of(index);
                 })
            .def("has_zero_size",
                 [](RuntimeTiledTensorView<T> &self, py::args args) -> bool {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled TensorView to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.has_zero_size(cast_args);
                 })
            .def("has_zero_size",
                 [](RuntimeTiledTensorView<T> &self, std::vector<ptrdiff_t> const &index) -> bool {
                     if (index.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of indices passed to tiled TensorView to check tile! Expected {}, got {}.",
                                                 self.rank(), index.size());
                     }
                     return self.has_zero_size(index);
                 })
            .def("__getitem__",
                 [](RuntimeTiledTensorView<T> const &self, py::tuple args) -> T {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled TensorView to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     return self.at(cast_args);
                 })
            .def("__setitem__",
                 [](RuntimeTiledTensorView<T> &self, py::tuple args, T value) {
                     if (args.size() != self.rank()) {
                         EINSUMS_THROW_EXCEPTION(num_argument_error,
                                                 "Wrong number of arguments passed to tiled TensorView to check tile! Expected {}, got {}.",
                                                 self.rank(), args.size());
                     }

                     std::vector<ptrdiff_t> cast_args(self.rank());

                     for (size_t i = 0; i < self.rank(); i++) {
                         cast_args[i] = py::cast<ptrdiff_t>(args[i]);
                     }

                     self(cast_args) = value;
                 })
            .def("zero", &RuntimeTiledTensorView<T>::zero)
            .def("zero_no_clear", &RuntimeTiledTensorView<T>::zero_no_clear)
            .def("set_all", &RuntimeTiledTensorView<T>::set_all)
            .def("set_all_existing", &RuntimeTiledTensorView<T>::set_all_existing)
            .def(py::self += T())
            .def(py::self -= T())
            .def(py::self *= T())
            .def(py::self /= T())
            .def(py::self += RuntimeTiledTensorView<T>())
            .def(py::self -= RuntimeTiledTensorView<T>())
            .def(py::self *= RuntimeTiledTensorView<T>())
            .def(py::self /= RuntimeTiledTensorView<T>())
            .def_property_readonly("tile_offsets", &RuntimeTiledTensorView<T>::tile_offsets)
            .def_property_readonly("tile_sizes", &RuntimeTiledTensorView<T>::tile_sizes)
            .def_property("name", &RuntimeTiledTensorView<T>::name, &RuntimeTiledTensorView<T>::set_name)
            .def("size", &RuntimeTiledTensorView<T>::size)
            .def("grid_size", static_cast<size_t (RuntimeTiledTensorView<T>::*)() const>(&RuntimeTiledTensorView<T>::grid_size))
            .def("grid_size", static_cast<size_t (RuntimeTiledTensorView<T>::*)(int) const>(&RuntimeTiledTensorView<T>::grid_size))
            .def("num_filled", &RuntimeTiledTensorView<T>::num_filled)
            .def("__len__", &RuntimeTiledTensorView<T>::num_filled)
            .def("full_view_of_underlying", &RuntimeTiledTensorView<T>::full_view_of_underlying)
            .def_property_readonly("shape", &RuntimeTiledTensorView<T>::dims)
            .def("insert_tile", static_cast<void (RuntimeTiledTensorView<T>::*)(std::vector<size_t> const &, RuntimeTensorView<T> const &)>(
                                    &RuntimeTiledTensorView<T>::insert_tile))
            .def("__iter__", [](RuntimeTiledTensorView<T> &self) { return py::make_iterator(self.tiles().begin(), self.tiles().end()); })
            .def("copy", [](RuntimeTiledTensorView<T> const &self) -> RuntimeTiledTensorView<T> { return self; })
            .def("__copy__", [](RuntimeTiledTensorView<T> const &self) -> RuntimeTiledTensorView<T> { return self; })
            .def("deepcopy", [](RuntimeTiledTensorView<T> const &self) -> RuntimeTiledTensorView<T> { return self; })
            .def("__deepcopy__", [](RuntimeTiledTensorView<T> const &self) -> RuntimeTiledTensorView<T> { return self; })
            .def("rank", &RuntimeTiledTensorView<T>::rank);
}

} // namespace einsums::python
