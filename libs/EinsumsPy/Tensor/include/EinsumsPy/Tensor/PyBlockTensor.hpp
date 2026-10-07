//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorUtilities.hpp>

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
using SharedRuntimeBlockTensor = std::shared_ptr<RuntimeBlockTensor<T>>;

template <typename T>
class PyBlockTensor : public RuntimeBlockTensor<T> {

    using typename RuntimeBlockTensor<T>::StoredType;

    using typename RuntimeBlockTensor<T>::ValueType;

    using RuntimeBlockTensor<T>::RuntimeBlockTensor;

    void zero() override { PYBIND11_OVERRIDE(void, RuntimeBlockTensor<T>, zero); }

    void set_all(T value) override { PYBIND11_OVERRIDE(void, RuntimeBlockTensor<T>, set_all, value); }

    StoredType const &block(int id) const override { PYBIND11_OVERRIDE(StoredType &, RuntimeBlockTensor<T>, block, id); }

    StoredType &block(int id) override { PYBIND11_OVERRIDE(StoredType &, RuntimeBlockTensor<T>, block, id); }

    StoredType const &block(std::string const &name) const override { PYBIND11_OVERRIDE(StoredType &, RuntimeBlockTensor<T>, block, name); }

    StoredType &block(std::string const &name) override { PYBIND11_OVERRIDE(StoredType &, RuntimeBlockTensor<T>, block, name); }

    void push_block(StoredType const &value) override { PYBIND11_OVERRIDE(void, RuntimeBlockTensor<T>, push_block, value); }

    void insert_block(int pos, StoredType const &value) override {
        PYBIND11_OVERRIDE(void, RuntimeBlockTensor<T>, insert_block, pos, value);
    }

  private:
    static void fill_with_subscript(pybind11::tuple const &args, std::vector<ptrdiff_t> *out) {
        out->resize(args.size());

        for (int i = 0; i < args.size(); i++) {
            auto const &arg = args[i];

            out->at(i) = pybind11::cast<ptrdiff_t>(arg);
        }
    }

  public:
    pybind11::object subscript(pybind11::tuple const &args) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeBlockTensor<T> *>(this), "__subscript");

        if (override) {
            auto o = override(args);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (args.size() < this->_rank) {
                EINSUMS_THROW_EXCEPTION(not_enough_args,
                                        "Not enough indices passed to block tensor! Views of block tensors not yet supported!");
            }

            if (args.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to block tensor!");
            }

            for (int i = 0; i < args.size(); i++) {
                auto const &arg = args[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    EINSUMS_THROW_EXCEPTION(
                        std::runtime_error,
                        "Can't handle slices while indexing a block tensor, as views of block tensors are not yet implemented!");
                }
            }

            std::vector<ptrdiff_t> index;

            fill_with_subscript(args, &index);

            return pybind11::cast((*this)(index));
        }
    }

    pybind11::object assign_values(T value, pybind11::tuple const &indices) {
        pybind11::gil_scoped_acquire gil;
        pybind11::function           override = pybind11::get_override(static_cast<RuntimeBlockTensor<T> *>(this), "__assign");

        if (override) {
            auto o = override(indices);
            if (pybind11::detail::cast_is_temporary_value_reference<pybind11::object>::value) {
                static pybind11::detail::override_caster_t<pybind11::object> caster;
                return pybind11::detail::cast_ref<pybind11::object>(std::move(o), caster);
            }
            return pybind11::detail::cast_safe<pybind11::object>(std::move(o));
        } else {
            if (indices.size() < this->_rank) {
                EINSUMS_THROW_EXCEPTION(not_enough_args,
                                        "Not enough indices passed to block tensor! Views of block tensors not yet supported!");
            }

            if (indices.size() > this->_rank) {
                EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to block tensor!");
            }

            for (int i = 0; i < indices.size(); i++) {
                auto const &arg = indices[i];

                if (pybind11::isinstance<pybind11::slice>(arg)) {
                    EINSUMS_THROW_EXCEPTION(
                        std::runtime_error,
                        "Can't handle slices while indexing a block tensor, as views of block tensors are not yet implemented!");
                }
            }

            std::vector<ptrdiff_t> index;

            fill_with_subscript(indices, &index);

            T &ret_ref = (*this)(index);

            ret_ref = value;

            return pybind11::cast(ret_ref);
        }
    }

#define OPERATOR(OP, NAME)                                                                                                                 \
    template <typename TOther>                                                                                                             \
    RuntimeBlockTensor<T> &operator OP(TOther const &other) {                                                                              \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, NAME, other);                                                              \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    RuntimeBlockTensor<T> &operator OP(RuntimeTensor<TOther> const &other) {                                                               \
        PYBIND11_OVERRIDE(RuntimeTensor<T> &, RuntimeTensor<T>, NAME, other);                                                              \
    }

    OPERATOR(=, operator=)
    OPERATOR(*=, operator*=)
    OPERATOR(/=, operator/=)
    OPERATOR(+=, operator+=)
    OPERATOR(-=, operator-=)

#undef OPERATOR
};

} // namespace einsums::python
