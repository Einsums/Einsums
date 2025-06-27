//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomSemidefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

namespace einsums::python {

namespace detail {
template <typename T, typename U>
void rdiv(RuntimeTensor<T> &out, U const &numerator) {
    size_t elems = out.size();

    auto *data = out.data();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (IsComplexV<U> && !IsComplexV<T>) {
            data[i] = std::real(numerator) / data[i];
        } else {
            data[i] = (T)numerator / data[i];
        }
    }
}

template <typename T, typename U>
void rdiv(RuntimeTensorView<T> &out, U const &numerator) {

    auto *data = out.data();

    std::vector<size_t> index_strides;

    auto strides = out.strides();

    size_t elems = dims_to_strides(out.dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        size_t data_sentinel;
        sentinel_to_sentinels(i, index_strides, strides, data_sentinel);

        if constexpr (IsComplexV<U> && !IsComplexV<T>) {
            data[data_sentinel] = std::real(numerator) / data[data_sentinel];
        } else {
            data[data_sentinel] = (T)numerator / data[data_sentinel];
        }
    }
}

template <typename T>
RuntimeTensor<T> transpose(RuntimeTensor<T> const &in) {

    if (in.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only transpose matrices.");
    }

    RuntimeTensor<T> out(in.name() + " transposed", {in.dim(1), in.dim(0)});

    TensorView<T, 2> in_view(in), out_view(out);

    einsums::tensor_algebra::permute(0.0, einsums::Indices{index::i, index::j}, &out_view, 1.0, einsums::Indices{index::j, index::i},
                                     in_view);

    return out;
}

template <typename T>
RuntimeTensor<T> transpose(RuntimeTensorView<T> const &in) {

    if (in.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only transpose matrices.");
    }

    RuntimeTensor<T> out(in.name() + " transposed", {in.dim(1), in.dim(0)});

    TensorView<T, 2> in_view(in), out_view(out);

    einsums::tensor_algebra::permute(0.0, einsums::Indices{index::i, index::j}, &out_view, 1.0, einsums::Indices{index::j, index::i},
                                     in_view);

    return out;
}

} // namespace detail

/**
 * @brief Expose runtime tensors to Python.
 *
 * @tparam T The stored type of the tensors to export.
 * @param mod The module which will contain the definitions.
 */
template <typename T>
void export_tensor(pybind11::module &mod) {
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

    mod.def(("create_random_tensor" + suffix).c_str(),
            [](std::string const &name, std::vector<size_t> const &dims) { return einsums::create_random_tensor<T>(name, dims); })
        .def(("create_random_definite" + suffix).c_str(),
             [](std::string const &name, size_t rows, RemoveComplexT<T> mean) {
                 return RuntimeTensor<T>(einsums::create_random_definite<T>(name, rows, rows));
             },
             pybind11::arg("name"), pybind11::arg("rows"), pybind11::arg("mean") = RemoveComplexT<T>{1.0})
        .def(("create_random_semidefinite" + suffix).c_str(),
             [](std::string const &name, size_t rows, RemoveComplexT<T> mean, int force_zeros) {
                 return RuntimeTensor<T>(einsums::create_random_semidefinite<T>(name, rows, rows));
             },
             pybind11::arg("name"), pybind11::arg("rows"), pybind11::arg("mean") = RemoveComplexT<T>{1.0},
             pybind11::arg("force_zeros") = 1);

    pybind11::class_<PyTensorIterator<T>, std::shared_ptr<PyTensorIterator<T>>>(mod, ("PyTensorIterator" + suffix).c_str())
        .def("__next__", &PyTensorIterator<T>::next, pybind11::return_value_policy::reference)
        .def("reversed", &PyTensorIterator<T>::reversed)
        .def("__iter__", [](PyTensorIterator<T> const &copy) { return copy; })
        .def("__reversed__", [](PyTensorIterator<T> const &copy) { return PyTensorIterator<T>(copy, true); });

    auto tensor_view =
        pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>, einsums::tensor_base::RuntimeTensorNoType>(
            mod, ("RuntimeTensorView" + suffix).c_str(), pybind11::buffer_protocol());
    pybind11::class_<RuntimeTensor<T>, PyTensor<T>, SharedRuntimeTensor<T>, einsums::tensor_base::RuntimeTensorNoType>(
        mod, ("RuntimeTensor" + suffix).c_str(), pybind11::buffer_protocol())
        .def(pybind11::init<>())
        .def(pybind11::init<std::string, std::vector<size_t> const &>())
        .def(pybind11::init<std::vector<size_t> const &>())
        .def(pybind11::init<pybind11::buffer const &>())
        .def("zero", &RuntimeTensor<T>::zero)
        .def("set_all", &RuntimeTensor<T>::set_all)
        .def("__getitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__getitem__",
             [](RuntimeTensor<T> &self, pybind11::slice const &args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__getitem__",
             [](RuntimeTensor<T> &self, int args) {
                 PyTensorView<T> cast(self);
                 return cast.subscript(args);
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, long key, double value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>) value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T) value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, long key, long value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T)value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, long key, std::complex<double> value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{value}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((RemoveComplexT<T>)value.real(), pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::slice const &key, double value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>) value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T) value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::slice const &key, long value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T)value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::slice const &key, std::complex<double> value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{value}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((RemoveComplexT<T>)value.real(), pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &key, double value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>) value, 0.0}, key);
                 } else {
                    cast.assign_values((T) value, key);
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &key, long value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, key);
                 } else {
                    cast.assign_values((T)value, key);
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &key, std::complex<double> value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{value}, key);
                 } else {
                    cast.assign_values((RemoveComplexT<T>)value.real(), key);
                 }
             })
        .def("__setitem__",
             [](RuntimeTensor<T> &self, pybind11::tuple const &key, pybind11::buffer const &values) {
                 PyTensorView<T> cast(self);
                 cast.assign_values(values, key);
             })
#define OPERATOR(OP, TYPE) .def(pybind11::self OP RuntimeTensor<TYPE>()).def(pybind11::self OP RuntimeTensorView<TYPE>())

            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
        .def(pybind11::self *= long())
        .def(
            "__imul__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self *= other; },
            pybind11::is_operator()) OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>)
            OPERATOR(/=, std::complex<double>)
        .def(pybind11::self /= long())
        .def(
            "__itruediv__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self /= other; },
            pybind11::is_operator()) OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>)
            OPERATOR(+=, std::complex<double>)
        .def(pybind11::self += long())
        .def(
            "__iadd__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self += other; },
            pybind11::is_operator()) OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>)
            OPERATOR(-=, std::complex<double>)
        .def(pybind11::self -= long())
        .def(
            "__isub__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self -= other; },
            pybind11::is_operator())
#undef OPERATOR
#define OPERATOR(OP, TYPE) .def(pybind11::self OP TYPE())
            OPERATOR(*=, double)OPERATOR(*=, std::complex<double>)
            OPERATOR(+=, double)OPERATOR(+=, std::complex<double>)
            OPERATOR(-=, double)OPERATOR(-=, std::complex<double>)
            OPERATOR(/=, double)OPERATOR(/=, std::complex<double>)
#undef OPERATOR

#define OPERATOR(OPNAME, OP, TYPE)                                                                                                         \
    .def(                                                                                                                                  \
        OPNAME,                                                                                                                            \
        [](RuntimeTensor<T> const &self, RuntimeTensor<TYPE> const &other) {                                                               \
            RuntimeTensor<T> out(self);                                                                                                    \
            out OP           other;                                                                                                        \
            return out;                                                                                                                    \
        },                                                                                                                                 \
        pybind11::is_operator())                                                                                                           \
        .def(                                                                                                                              \
            OPNAME,                                                                                                                        \
            [](RuntimeTensor<T> const &self, RuntimeTensorView<TYPE> const &other) {                                                       \
                RuntimeTensor<T> out(self);                                                                                                \
                out OP           other;                                                                                                    \
                return out;                                                                                                                \
            },                                                                                                                             \
            pybind11::is_operator())

        OPERATOR("__mul__", *=, float)
        OPERATOR("__mul__", *=, double)
        OPERATOR("__mul__", *=, std::complex<float>)
        OPERATOR("__mul__", *=, std::complex<double>)
        OPERATOR("__truediv__", /=, float)
        OPERATOR("__truediv__", /=, double)
        OPERATOR("__truediv__", /=, std::complex<float>)
        OPERATOR("__truediv__", /=, std::complex<double>)
        OPERATOR("__add__", +=, float)
        OPERATOR("__add__", +=, double)
        OPERATOR("__add__", +=, std::complex<float>)
        OPERATOR("__add__", +=, std::complex<double>)
        OPERATOR("__sub__", -=, float)
        OPERATOR("__sub__", -=, double)
        OPERATOR("__sub__", -=, std::complex<float>)
        OPERATOR("__sub__", -=, std::complex<double>)
        OPERATOR("__rmul__", *=, float)
        OPERATOR("__rmul__", *=, double)
        OPERATOR("__rmul__", *=, std::complex<float>)
        OPERATOR("__rmul__", *=, std::complex<double>)
        OPERATOR("__radd__", +=, float)
        OPERATOR("__radd__", +=, double)
        OPERATOR("__radd__", +=, std::complex<float>)
        OPERATOR("__radd__", +=, std::complex<double>)

#undef OPERATOR

#define OPERATOR(OPNAME, OP, TYPE)                                                                                                         \
    .def(                                                                                                                                  \
        OPNAME,                                                                                                                            \
        [](RuntimeTensor<T> const &self, TYPE const &other) {                                                                              \
            RuntimeTensor<T> out(self);                                                                                                    \
            out OP           other;                                                                                                        \
            return out;                                                                                                                    \
        },                                                                                                                                 \
        pybind11::is_operator()) 
        OPERATOR("__mul__", *=, double)
        OPERATOR("__mul__", *=, std::complex<double>)
        OPERATOR("__truediv__", /=, double)
        OPERATOR("__truediv__", /=, std::complex<double>)
        OPERATOR("__add__", +=, double)
        OPERATOR("__add__", +=, std::complex<double>)
        OPERATOR("__sub__", -=, double)
        OPERATOR("__sub__", -=, std::complex<double>)
        OPERATOR("__rmul__", *=, double)
        OPERATOR("__rmul__", *=, std::complex<double>)
        OPERATOR("__radd__", +=, double)
        OPERATOR("__radd__", +=, std::complex<double>)
        .def("__mul__", [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out *=           other;
            return out;
        }, pybind11::is_operator())
        .def("__truediv__", [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out /=           other;
            return out;
        }, pybind11::is_operator())
        .def("__add__", [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out +=           other;
            return out;
        }, pybind11::is_operator())
        .def("__sub__", [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out -=           other;
            return out;
        }, pybind11::is_operator())
        .def("__mul__", [](RuntimeTensor<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out *=           other;
            return out;
        }, pybind11::is_operator())
        .def("__truediv__", [](RuntimeTensor<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out /=           other;
            return out;
        }, pybind11::is_operator())
        .def("__add__", [](RuntimeTensor<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out +=           other;
            return out;
        }, pybind11::is_operator())
        .def("__sub__", [](RuntimeTensor<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out -=           other;
            return out;
        }, pybind11::is_operator())

        .def("__rsub__", [](RuntimeTensor<T> const &self, double other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            }, pybind11 ::is_operator())
        .def("__rsub__", [](RuntimeTensor<T> const &self, std::complex<double> other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            }, pybind11 ::is_operator())
        .def("__rsub__", [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
                PyTensor<T> out(other);
                out -= self;
                return out;
            }, pybind11 ::is_operator())
        .def("__rsub__", [](RuntimeTensor<T> const &self, long other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensor<T> const &self, double other) {
                RuntimeTensor<T> out(self);
                detail::rdiv(out, other);
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensor<T> const &self, std::complex<double> other) {
                RuntimeTensor<T> out(self);
                detail::rdiv(out, other);
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
                PyTensor<T> out(other);
                out /= self;
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensor<T> const &self, long other) {
                RuntimeTensor<T> out(self);
                detail::rdiv(out, other);
                return out;
            }, pybind11 ::is_operator())
        
        .def("assign", [](PyTensor<T> &self, pybind11::buffer &buffer) { return self = buffer; })
        .def("dim", &RuntimeTensor<T>::dim)
        .def("dims", &RuntimeTensor<T>::dims)
        .def("stride", &RuntimeTensor<T>::stride)
        .def("strides", &RuntimeTensor<T>::strides)
        .def("to_rank_1_view", &RuntimeTensor<T>::to_rank_1_view)
        .def("get_name", &RuntimeTensor<T>::name)
        .def("set_name", &RuntimeTensor<T>::set_name)
        .def_property("name", &RuntimeTensor<T>::name, &RuntimeTensor<T>::set_name)
        .def_property_readonly("shape", [](RuntimeTensor<T> &self) { return pybind11::cast(self.dims()); })
        .def("size", &RuntimeTensor<T>::size)
        .def("__len__", &RuntimeTensor<T>::size)
        .def("__iter__", [](RuntimeTensor<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](RuntimeTensor<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensor<T>::rank)
        .def("__copy__", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__deepcopy__", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("copy", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("deepcopy", [](RuntimeTensor<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__str__",
             [](RuntimeTensor<T> const &self) {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def_property_readonly("T", [](RuntimeTensor<T> const &self) { return detail::transpose(self); })
        .def_buffer([](RuntimeTensor<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
    tensor_view.def(pybind11::init<>())
        .def(pybind11::init<RuntimeTensor<T> &>())
        .def(pybind11::init<RuntimeTensor<T> const &>())
        .def(pybind11::init<RuntimeTensorView<T> const &>())
        .def(pybind11::init<RuntimeTensor<T> &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensor<T> const &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensorView<T> &, std::vector<size_t> const &>())
        .def(pybind11::init<RuntimeTensorView<T> const &, std::vector<size_t> const &>())
        .def(pybind11::init<pybind11::buffer &>())
        .def("zero", &RuntimeTensorView<T>::zero)
        .def("set_all", &RuntimeTensorView<T>::set_all)
        .def("__getitem__", [](PyTensorView<T> &self, pybind11::tuple const &args) { return self.subscript(args); })
        .def("__getitem__", [](PyTensorView<T> &self, pybind11::slice const &args) { return self.subscript(args); })
        .def("__getitem__", [](PyTensorView<T> &self, int args) { return self.subscript(args); })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, long key, double value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>) value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T) value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, long key, long value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T)value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, long key, std::complex<double> value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{value}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((RemoveComplexT<T>)value.real(), pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, pybind11::slice const &key, double value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>) value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T) value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, pybind11::slice const &key, long value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((T)value, pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, pybind11::slice const &key, std::complex<double> value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{value}, pybind11::make_tuple(key));
                 } else {
                    cast.assign_values((RemoveComplexT<T>)value.real(), pybind11::make_tuple(key));
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, pybind11::tuple const &key, long value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, key);
                 } else {
                    cast.assign_values((T)value, key);
                 }
             })
        .def("__setitem__",
             [](RuntimeTensorView<T> &self, pybind11::tuple const &key, std::complex<double> value) {
                 PyTensorView<T> cast(self);
                 if constexpr (IsComplexV<T>) {
                     cast.assign_values(T{value}, key);
                 } else {
                    cast.assign_values((RemoveComplexT<T>)value.real(), key);
                 }
             })
        .def("__setitem__",
             [](PyTensorView<T> &self, pybind11::tuple const &key, pybind11::buffer const &values) { self.assign_values(values, key); })
#undef OPERATOR
#define OPERATOR(OP, TYPE) .def(pybind11::self OP RuntimeTensor<TYPE>()).def(pybind11::self OP RuntimeTensorView<TYPE>())
            OPERATOR(*=, float) OPERATOR(*=, double) OPERATOR(*=, std::complex<float>) OPERATOR(*=, std::complex<double>)
        .def(pybind11::self *= long())
        .def(
            "__imul__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self *= other; }, pybind11::is_operator())
            OPERATOR(/=, float) OPERATOR(/=, double) OPERATOR(/=, std::complex<float>) OPERATOR(/=, std::complex<double>)
        .def(pybind11::self /= long())
        .def(
            "__itruediv__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self /= other; }, pybind11::is_operator())
            OPERATOR(+=, float) OPERATOR(+=, double) OPERATOR(+=, std::complex<float>) OPERATOR(+=, std::complex<double>)
        .def(pybind11::self += long())
        .def(
            "__iadd__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self += other; }, pybind11::is_operator())
            OPERATOR(-=, float) OPERATOR(-=, double) OPERATOR(-=, std::complex<float>) OPERATOR(-=, std::complex<double>)
        .def(pybind11::self -= long())
        .def(
            "__isub__", [](PyTensorView<T> &self, pybind11::buffer const &other) { return self -= other; }, pybind11::is_operator())
#undef OPERATOR
#define OPERATOR(OP, TYPE) .def(pybind11::self OP TYPE())
            OPERATOR(*=, double)OPERATOR(*=, std::complex<double>)
            OPERATOR(+=, double)OPERATOR(+=, std::complex<double>)
            OPERATOR(-=, double)OPERATOR(-=, std::complex<double>)
            OPERATOR(/=, double)OPERATOR(/=, std::complex<double>)
#undef OPERATOR
#define OPERATOR(OPNAME, OP, TYPE)                                                                                                         \
    .def(                                                                                                                                  \
        OPNAME,                                                                                                                            \
        [](RuntimeTensorView<T> const &self, RuntimeTensor<TYPE> const &other) {                                                           \
            RuntimeTensor<T> out(self);                                                                                                    \
            out OP           other;                                                                                                        \
            return out;                                                                                                                    \
        },                                                                                                                                 \
        pybind11::is_operator())                                                                                                           \
        .def(                                                                                                                              \
            OPNAME,                                                                                                                        \
            [](RuntimeTensorView<T> const &self, RuntimeTensorView<TYPE> const &other) {                                                   \
                RuntimeTensor<T> out(self);                                                                                                \
                out OP           other;                                                                                                    \
                return out;                                                                                                                \
            },                                                                                                                             \
            pybind11::is_operator())

        OPERATOR("__mul__", *=, float)
        OPERATOR("__mul__", *=, double)
        OPERATOR("__mul__", *=, std::complex<float>)
        OPERATOR("__mul__", *=, std::complex<double>)
        OPERATOR("__truediv__", /=, float)
        OPERATOR("__truediv__", /=, double)
        OPERATOR("__truediv__", /=, std::complex<float>)
        OPERATOR("__truediv__", /=, std::complex<double>)
        OPERATOR("__add__", +=, float)
        OPERATOR("__add__", +=, double)
        OPERATOR("__add__", +=, std::complex<float>)
        OPERATOR("__add__", +=, std::complex<double>)
        OPERATOR("__sub__", -=, float)
        OPERATOR("__sub__", -=, double)
        OPERATOR("__sub__", -=, std::complex<float>)
        OPERATOR("__sub__", -=, std::complex<double>)
        OPERATOR("__rmul__", *=, float)
        OPERATOR("__rmul__", *=, double)
        OPERATOR("__rmul__", *=, std::complex<float>)
        OPERATOR("__rmul__", *=, std::complex<double>)
        OPERATOR("__radd__", +=, float)
        OPERATOR("__radd__", +=, double)
        OPERATOR("__radd__", +=, std::complex<float>)
        OPERATOR("__radd__", +=, std::complex<double>)
#undef OPERATOR
#define OPERATOR(OPNAME, OP, TYPE)                                                                                                         \
    .def(                                                                                                                                  \
        OPNAME,                                                                                                                            \
        [](RuntimeTensorView<T> const &self, TYPE const &other) {                                                                          \
            RuntimeTensor<T> out(self);                                                                                                    \
            out OP           other;                                                                                                        \
            return out;                                                                                                                    \
        },                                                                                                                                 \
        pybind11::is_operator())

        OPERATOR("__mul__", *=, double)
        OPERATOR("__mul__", *=, std::complex<double>)
        OPERATOR("__truediv__", /=, double)
        OPERATOR("__truediv__", /=, std::complex<double>)
        OPERATOR("__add__", +=, double)
        OPERATOR("__add__", +=, std::complex<double>)
        OPERATOR("__sub__", -=, double)
        OPERATOR("__sub__", -=, std::complex<double>)
        OPERATOR("__rmul__", *=, double)
        OPERATOR("__rmul__", *=, std::complex<double>)
        OPERATOR("__radd__", +=, double)
        OPERATOR("__radd__", +=, std::complex<double>)
        .def("__mul__", [](RuntimeTensorView<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out *=           other;
            return out;
        }, pybind11::is_operator())
        .def("__truediv__", [](RuntimeTensorView<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out /=           other;
            return out;
        }, pybind11::is_operator())
        .def("__add__", [](RuntimeTensorView<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out +=           other;
            return out;
        }, pybind11::is_operator())
        .def("__sub__", [](RuntimeTensorView<T> const &self, long other) {
            RuntimeTensor<T> out(self);                                                                                                   
            out -=           other;
            return out;
        }, pybind11::is_operator())
        .def("__mul__", [](RuntimeTensorView<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out *=           other;
            return out;
        }, pybind11::is_operator())
        .def("__truediv__", [](RuntimeTensorView<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out /=           other;
            return out;
        }, pybind11::is_operator())
        .def("__add__", [](RuntimeTensorView<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out +=           other;
            return out;
        }, pybind11::is_operator())
        .def("__sub__", [](RuntimeTensorView<T> const &self, pybind11::buffer const & other) {
            PyTensor<T> out(self);                                                                                                   
            out -=           other;
            return out;
        }, pybind11::is_operator())
        .def("__rsub__", [](RuntimeTensorView<T> const &self, double other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            }, pybind11 ::is_operator())
        .def("__rsub__", [](RuntimeTensorView<T> const &self, std::complex<double> other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            }, pybind11 ::is_operator())
        .def("__rsub__", [](RuntimeTensorView<T> const &self, pybind11::buffer const &other) {
                PyTensor<T> out(other);
                out -= self;
                return out;
            }, pybind11 ::is_operator())
        .def("__rsub__", [](RuntimeTensorView<T> const &self, long other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensorView<T> const &self, double other) {
                RuntimeTensor<T> out(self);
                detail::rdiv(out, other);
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensorView<T> const &self, std::complex<double> other) {
                RuntimeTensor<T> out(self);
                detail::rdiv(out, other);
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensorView<T> const &self, pybind11::buffer const &other) {
                PyTensor<T> out(other);
                out /= self;
                return out;
            }, pybind11 ::is_operator())
        .def("__rdiv__", [](RuntimeTensorView<T> const &self, long other) {
                RuntimeTensor<T> out(self);
                detail::rdiv(out, other);
                return out;
            }, pybind11 ::is_operator())
        .def("assign", [](PyTensorView<T> &self, pybind11::buffer &buffer) { return self = buffer; })
        .def("dim", &RuntimeTensorView<T>::dim)
        .def("dims", &RuntimeTensorView<T>::dims)
        .def("stride", &RuntimeTensorView<T>::stride)
        .def("strides", &RuntimeTensorView<T>::strides)
        .def("get_name", &RuntimeTensorView<T>::name)
        .def("set_name", &RuntimeTensorView<T>::set_name)
        .def_property("name", &RuntimeTensorView<T>::name, &RuntimeTensorView<T>::set_name)
        .def_property_readonly("shape", [](RuntimeTensorView<T> &self) { return pybind11::cast(self.dims()); })
        .def("size", &RuntimeTensorView<T>::size)
        .def("copy", [](RuntimeTensorView<T> const &self) { return RuntimeTensor<T>(self); })
        .def("__len__", &RuntimeTensorView<T>::size)
        .def("__iter__", [](RuntimeTensorView<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor); })
        .def("__reversed__", [](RuntimeTensorView<T> const &tensor) { return std::make_shared<PyTensorIterator<T>>(tensor, true); })
        .def("rank", &RuntimeTensorView<T>::rank)
        .def("__str__",
             [](RuntimeTensorView<T> const &self) {
                 std::stringstream stream;
                 fprintln(stream, self);
                 return stream.str();
             })
        .def_property_readonly("T", [](RuntimeTensorView<T> const &self) { return detail::transpose(self); })
        .def_buffer([](RuntimeTensorView<T> &self) {
            std::vector<ptrdiff_t> dims(self.rank()), strides(self.rank());
            for (int i = 0; i < self.rank(); i++) {
                dims[i]    = self.dim(i);
                strides[i] = sizeof(T) * self.stride(i);
            }

            return pybind11::buffer_info(self.data(), sizeof(T), pybind11::format_descriptor<T>::format(), self.rank(), dims, strides);
        });
#undef OPERATOR
}
} // namespace einsums::python