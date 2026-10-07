//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomSemidefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <EinsumsPy/Tensor/PyBlockTensor.hpp>
#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <cstdlib>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace einsums::python {

namespace detail {
EINSUMS_EXPORT void fill_with_subscript(pybind11::tuple const &args, std::vector<ptrdiff_t> *out);

} // namespace detail

template <typename T>
void export_block_tensor(pybind11::module &mod) {
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

    //    auto x = detail::resolve_func<int (RuntimeBlockTensor<T>::*)(size_t) const>(&RuntimeBlockTensor<T>::block_of);

    pybind11::class_<RuntimeBlockTensor<T>, PyBlockTensor<T>, SharedRuntimeBlockTensor<T>, tensor_base::BlockTensorNoExtra>(
        mod, ("BlockTensor" + suffix).c_str())
        .def(pybind11::init<size_t>())
        .def(pybind11::init<std::string, size_t, std::vector<size_t>>())
        .def(pybind11::init<size_t, std::vector<size_t>>())
        .def(pybind11::init<std::string, size_t>())
        .def("zero", &RuntimeBlockTensor<T>::zero)
        .def("set_all", &RuntimeBlockTensor<T>::set_all)
        .def("block", static_cast<RuntimeTensor<T> &(RuntimeBlockTensor<T>::*)(int)>(&RuntimeBlockTensor<T>::block),
             pybind11::return_value_policy::reference_internal)
        .def("block", static_cast<RuntimeTensor<T> &(RuntimeBlockTensor<T>::*)(std::string const &)>(&RuntimeBlockTensor<T>::block),
             pybind11::return_value_policy::reference_internal)
        .def("push_block", &RuntimeBlockTensor<T>::push_block)
        .def("push_block", [](RuntimeBlockTensor<T> &self, pybind11::buffer const &buffer) { self.push_block(PyTensor<T>(buffer)); })
        .def("insert_block", &RuntimeBlockTensor<T>::insert_block)
        .def("insert_block",
             [](RuntimeBlockTensor<T> &self, long index, pybind11::buffer const &buffer) { self.insert_block(index, PyTensor<T>(buffer)); })
        .def("is_inside_block",
             [](RuntimeBlockTensor<T> const &self, std::vector<long> const &index) -> bool { return self.is_inside_block(index); })
        .def("is_inside_block",
             [](RuntimeBlockTensor<T> const &self, pybind11::args args) -> bool {
                 std::vector<ptrdiff_t> index;
                 detail::fill_with_subscript(args, &index);
                 return self.is_inside_block(index);
             })
        .def("__getitem__",
             [](RuntimeBlockTensor<T> const &self, pybind11::tuple const &args) -> T {
                 std::vector<ptrdiff_t> index(args.size());

                 detail::fill_with_subscript(args, &index);

                 return self(index);
             })
        .def("__setitem__",
             [](RuntimeBlockTensor<T> &self, pybind11::tuple const &position, T value) {
                 std::vector<ptrdiff_t> index(position.size());

                 detail::fill_with_subscript(position, &index);

                 self(index) = value;
             })
#define OPERATOR(OP, TYPE) .def(pybind11::self OP RuntimeBlockTensor<TYPE>()).def(pybind11::self OP TYPE())
            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
                OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>) OPERATOR(/=, std::complex<double>)
                    OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>) OPERATOR(+=, std::complex<double>)
                        OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>) OPERATOR(-=, std::complex<double>)
#undef OPERATOR
        .def(pybind11::self *= long())
        .def(pybind11::self /= long())
        .def(pybind11::self += long())
        .def(pybind11::self -= long())
        .def("num_blocks", &RuntimeBlockTensor<T>::num_blocks)
        .def("__len__", &RuntimeBlockTensor<T>::num_blocks)
        .def("block_dims", static_cast<std::vector<size_t> const &(RuntimeBlockTensor<T>::*)() const>(&RuntimeBlockTensor<T>::block_dims),
             pybind11::return_value_policy::copy)
        .def("block_dims", static_cast<std::vector<size_t> (RuntimeBlockTensor<T>::*)(size_t) const>(&RuntimeBlockTensor<T>::block_dims))
        .def("block_dims",
             static_cast<std::vector<size_t> (RuntimeBlockTensor<T>::*)(std::string const &) const>(&RuntimeBlockTensor<T>::block_dims))
        .def("ranges", &RuntimeBlockTensor<T>::ranges)
        .def("block_range", &RuntimeBlockTensor<T>::block_range)
        .def("block_dim", static_cast<size_t (RuntimeBlockTensor<T>::*)(size_t, int) const>(&RuntimeBlockTensor<T>::block_dim),
             pybind11::arg("block"), pybind11::arg("ind") = 0)
        .def("block_dim", static_cast<size_t (RuntimeBlockTensor<T>::*)(std::string const &, int) const>(&RuntimeBlockTensor<T>::block_dim),
             pybind11::arg("name"), pybind11::arg("ind") = 0)
        .def("dim", static_cast<size_t (RuntimeBlockTensor<T>::*)() const>(&RuntimeBlockTensor<T>::dim))
        .def("dim", static_cast<size_t (RuntimeBlockTensor<T>::*)(int) const>(&RuntimeBlockTensor<T>::dim))
        .def("blocks", static_cast<std::vector<RuntimeTensor<T>> &(RuntimeBlockTensor<T>::*)()>(&RuntimeBlockTensor<T>::vector_data),
             pybind11::return_value_policy::reference_internal)
        .def(
            "__iter__",
            [](RuntimeBlockTensor<T> &self) { return pybind11::make_iterator(self.vector_data().begin(), self.vector_data().end()); },
            pybind11::keep_alive<0, 1>())
        .def_property("name", static_cast<std::string const &(RuntimeBlockTensor<T>::*)() const>(&RuntimeBlockTensor<T>::name),
                      static_cast<void (RuntimeBlockTensor<T>::*)(std::string const &)>(&RuntimeBlockTensor<T>::set_name),
                      pybind11::return_value_policy::copy)
        .def("size", &RuntimeBlockTensor<T>::size)
        .def("rank", &RuntimeBlockTensor<T>::rank)
        .def("__str__",
             [](RuntimeBlockTensor<T> const &self) -> std::string {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def("__copy__", [](RuntimeBlockTensor<T> const &self) { return RuntimeBlockTensor<T>(self); })
        .def("__deepcopy__", [](RuntimeBlockTensor<T> const &self) { return RuntimeBlockTensor<T>(self); })
        .def("copy", [](RuntimeBlockTensor<T> const &self) { return RuntimeBlockTensor<T>(self); })
        .def("deepcopy", [](RuntimeBlockTensor<T> const &self) { return RuntimeBlockTensor<T>(self); });
}

} // namespace einsums::python