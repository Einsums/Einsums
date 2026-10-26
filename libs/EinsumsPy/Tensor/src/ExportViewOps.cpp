//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <EinsumsPy/Tensor/PyTensor.hpp>
#include <EinsumsPy/Tensor/TensorExport.hpp>

namespace einsums::python {
template <typename T>
void export_tensor_view_ops(detail::ExportTensorViewClass<T> &tensor) {
    tensor    .def("zero", &RuntimeTensorView<T>::zero)
            .def("set_all", &RuntimeTensorView<T>::set_all)    .def("__getitem__", [](PyTensorView<T> &self, pybind11::tuple const &args) { return self.subscript(args); })
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
                 [](PyTensorView<T> &self, pybind11::slice const &key, pybind11::buffer const &values) { self.assign_values(values, key); })
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
            .def("assign", [](PyTensorView<T> &self, pybind11::buffer &buffer) { return self = buffer; });
#undef OPERATOR
}

void export_tensor_view_opsf(detail::ExportTensorViewClass<float> &tensor) {
    export_tensor_view_ops<float>(tensor);
}

void export_tensor_view_opsd(detail::ExportTensorViewClass<double> &tensor) {
    export_tensor_view_ops<double>(tensor);
}

void export_tensor_view_opsc(detail::ExportTensorViewClass<std::complex<float>> &tensor) {
    export_tensor_view_ops<std::complex<float>>(tensor);
}

void export_tensor_view_opsz(detail::ExportTensorViewClass<std::complex<double>> &tensor) {
    export_tensor_view_ops<std::complex<double>>(tensor);
}

} // namespace einsums::python