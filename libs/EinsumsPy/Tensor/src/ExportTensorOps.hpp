//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_ops(detail::ExportTensorClass<T> &tensor) {
    tensor.def("zero", &RuntimeTensor<T>::zero).def("set_all", &RuntimeTensor<T>::set_all);
    tensor.def("__getitem__", [](RuntimeTensor<T> &self, pybind11::tuple const &args) {
        PyTensorView<T> cast(self);
        return cast.subscript(args);
    });
    tensor.def("__getitem__", [](RuntimeTensor<T> &self, pybind11::slice const &args) {
        PyTensorView<T> cast(self);
        return cast.subscript(args);
    });
    tensor.def("__getitem__", [](RuntimeTensor<T> &self, int args) {
        PyTensorView<T> cast(self);
        return cast.subscript(args);
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, long key, double value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
        } else {
            cast.assign_values((T)value, pybind11::make_tuple(key));
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, long key, long value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
        } else {
            cast.assign_values((T)value, pybind11::make_tuple(key));
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, long key, std::complex<double> value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{value}, pybind11::make_tuple(key));
        } else {
            cast.assign_values((RemoveComplexT<T>)value.real(), pybind11::make_tuple(key));
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::slice const &key, double value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
        } else {
            cast.assign_values((T)value, pybind11::make_tuple(key));
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::slice const &key, long value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, pybind11::make_tuple(key));
        } else {
            cast.assign_values((T)value, pybind11::make_tuple(key));
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::slice const &key, std::complex<double> value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{value}, pybind11::make_tuple(key));
        } else {
            cast.assign_values((RemoveComplexT<T>)value.real(), pybind11::make_tuple(key));
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::slice const &key, pybind11::buffer const &values) {
        PyTensorView<T> cast(self);
        cast.assign_values(values, key);
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::tuple const &key, double value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, key);
        } else {
            cast.assign_values((T)value, key);
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::tuple const &key, long value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{(RemoveComplexT<T>)value, 0.0}, key);
        } else {
            cast.assign_values((T)value, key);
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::tuple const &key, std::complex<double> value) {
        PyTensorView<T> cast(self);
        if constexpr (IsComplexV<T>) {
            cast.assign_values(T{value}, key);
        } else {
            cast.assign_values((RemoveComplexT<T>)value.real(), key);
        }
    });
    tensor.def("__setitem__", [](RuntimeTensor<T> &self, pybind11::tuple const &key, pybind11::buffer const &values) {
        PyTensorView<T> cast(self);
        cast.assign_values(values, key);
    });
#define OPERATOR(OP, TYPE) tensor.def(pybind11::self OP RuntimeTensor<TYPE>()).def(pybind11::self OP RuntimeTensorView<TYPE>())

    OPERATOR(*=, float);
    OPERATOR(*=, double);
    OPERATOR(*=, std::complex<float>);
    OPERATOR(*=, std::complex<double>);
    tensor.def(pybind11::self *= long());
    tensor.def(
        "__imul__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self *= other; },
        pybind11::is_operator());
    OPERATOR(/=, float);
    OPERATOR(/=, double);
    OPERATOR(/=, std::complex<float>);
    OPERATOR(/=, std::complex<double>);
    tensor.def(pybind11::self /= long());
    tensor.def(
        "__itruediv__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self /= other; },
        pybind11::is_operator());
    OPERATOR(+=, float);
    OPERATOR(+=, double);
    OPERATOR(+=, std::complex<float>);
    OPERATOR(+=, std::complex<double>);
    tensor.def(pybind11::self += long());
    tensor.def(
        "__iadd__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self += other; },
        pybind11::is_operator());
    OPERATOR(-=, float);
    OPERATOR(-=, double);
    OPERATOR(-=, std::complex<float>);
    OPERATOR(-=, std::complex<double>);
    tensor.def(pybind11::self -= long());
    tensor.def(
        "__isub__", [](PyTensor<T> &self, pybind11::buffer const &other) -> RuntimeTensor<T> & { return self -= other; },
        pybind11::is_operator());
#undef OPERATOR
#define OPERATOR(OP, TYPE) tensor.def(pybind11::self OP TYPE())
    OPERATOR(*=, double);
    OPERATOR(*=, std::complex<double>);
    OPERATOR(+=, double);
    OPERATOR(+=, std::complex<double>);
    OPERATOR(-=, double);
    OPERATOR(-=, std::complex<double>);
    OPERATOR(/=, double);
    OPERATOR(/=, std::complex<double>);
#undef OPERATOR

#define OPERATOR(OPNAME, OP, TYPE)                                                                                                         \
    tensor                                                                                                                                 \
        .def(                                                                                                                              \
            OPNAME,                                                                                                                        \
            [](RuntimeTensor<T> const &self, RuntimeTensor<TYPE> const &other) {                                                           \
                RuntimeTensor<T> out(self);                                                                                                \
                out OP           other;                                                                                                    \
                return out;                                                                                                                \
            },                                                                                                                             \
            pybind11::is_operator())                                                                                                       \
        .def(                                                                                                                              \
            OPNAME,                                                                                                                        \
            [](RuntimeTensor<T> const &self, RuntimeTensorView<TYPE> const &other) {                                                       \
                RuntimeTensor<T> out(self);                                                                                                \
                out OP           other;                                                                                                    \
                return out;                                                                                                                \
            },                                                                                                                             \
            pybind11::is_operator())

    OPERATOR("__mul__", *=, float);
    OPERATOR("__mul__", *=, double);
    OPERATOR("__mul__", *=, std::complex<float>);
    OPERATOR("__mul__", *=, std::complex<double>);
    OPERATOR("__truediv__", /=, float);
    OPERATOR("__truediv__", /=, double);
    OPERATOR("__truediv__", /=, std::complex<float>);
    OPERATOR("__truediv__", /=, std::complex<double>);
    OPERATOR("__add__", +=, float);
    OPERATOR("__add__", +=, double);
    OPERATOR("__add__", +=, std::complex<float>);
    OPERATOR("__add__", +=, std::complex<double>);
    OPERATOR("__sub__", -=, float);
    OPERATOR("__sub__", -=, double);
    OPERATOR("__sub__", -=, std::complex<float>);
    OPERATOR("__sub__", -=, std::complex<double>);
    OPERATOR("__rmul__", *=, float);
    OPERATOR("__rmul__", *=, double);
    OPERATOR("__rmul__", *=, std::complex<float>);
    OPERATOR("__rmul__", *=, std::complex<double>);
    OPERATOR("__radd__", +=, float);
    OPERATOR("__radd__", +=, double);
    OPERATOR("__radd__", +=, std::complex<float>);
    OPERATOR("__radd__", +=, std::complex<double>);

#undef OPERATOR

#define OPERATOR(OPNAME, OP, TYPE)                                                                                                         \
    tensor.def(                                                                                                                            \
        OPNAME,                                                                                                                            \
        [](RuntimeTensor<T> const &self, TYPE const &other) {                                                                              \
            RuntimeTensor<T> out(self);                                                                                                    \
            out OP           other;                                                                                                        \
            return out;                                                                                                                    \
        },                                                                                                                                 \
        pybind11::is_operator())
    OPERATOR("__mul__", *=, double);
    OPERATOR("__mul__", *=, std::complex<double>);
    OPERATOR("__truediv__", /=, double);
    OPERATOR("__truediv__", /=, std::complex<double>);
    OPERATOR("__add__", +=, double);
    OPERATOR("__add__", +=, std::complex<double>);
    OPERATOR("__sub__", -=, double);
    OPERATOR("__sub__", -=, std::complex<double>);
    OPERATOR("__rmul__", *=, double);
    OPERATOR("__rmul__", *=, std::complex<double>);
    OPERATOR("__radd__", +=, double);
    OPERATOR("__radd__", +=, std::complex<double>);
    tensor.def(
        "__mul__",
        [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);
            out *= other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__truediv__",
        [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);
            out /= other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__add__",
        [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);
            out += other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__sub__",
        [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);
            out -= other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__mul__",
        [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
            PyTensor<T> out(self);
            out *= other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__truediv__",
        [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
            PyTensor<T> out(self);
            out /= other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__add__",
        [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
            PyTensor<T> out(self);
            out += other;
            return out;
        },
        pybind11::is_operator());
    tensor.def(
        "__sub__",
        [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
            PyTensor<T> out(self);
            out -= other;
            return out;
        },
        pybind11::is_operator());
    tensor

        .def(
            "__rsub__",
            [](RuntimeTensor<T> const &self, double other) {
                RuntimeTensor<T> out(self);
                out -= other;
                out *= T{-1.0};
                return out;
            },
            pybind11 ::is_operator());
    tensor.def(
        "__rsub__",
        [](RuntimeTensor<T> const &self, std::complex<double> other) {
            RuntimeTensor<T> out(self);
            out -= other;
            out *= T{-1.0};
            return out;
        },
        pybind11 ::is_operator());
    tensor.def(
        "__rsub__",
        [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
            PyTensor<T> out(other);
            out -= self;
            return out;
        },
        pybind11 ::is_operator());
    tensor.def(
        "__rsub__",
        [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);
            out -= other;
            out *= T{-1.0};
            return out;
        },
        pybind11 ::is_operator());
    tensor.def(
        "__rdiv__",
        [](RuntimeTensor<T> const &self, double other) {
            RuntimeTensor<T> out(self);
            detail::rdiv(out, other);
            return out;
        },
        pybind11 ::is_operator());
    tensor.def(
        "__rdiv__",
        [](RuntimeTensor<T> const &self, std::complex<double> other) {
            RuntimeTensor<T> out(self);
            detail::rdiv(out, other);
            return out;
        },
        pybind11 ::is_operator());
    tensor.def(
        "__rdiv__",
        [](RuntimeTensor<T> const &self, pybind11::buffer const &other) {
            PyTensor<T> out(other);
            out /= self;
            return out;
        },
        pybind11 ::is_operator());
    tensor.def(
        "__rdiv__",
        [](RuntimeTensor<T> const &self, long other) {
            RuntimeTensor<T> out(self);
            detail::rdiv(out, other);
            return out;
        },
        pybind11 ::is_operator());
    tensor.def("assign", [](PyTensor<T> &self, pybind11::buffer &buffer) { return self = buffer; });
#undef OPERATOR
}
} // namespace einsums::python
