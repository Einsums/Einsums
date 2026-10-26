//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

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

    BufferVector<size_t> index_strides;

    auto strides = out.strides();

    size_t elems = dims_to_strides(out.dims(), index_strides, true);

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

template <typename T>
using ExportTensorClass =
    pybind11::class_<einsums::GeneralRuntimeTensor<T, std::allocator<T>>, einsums::python::PyTensor<T>,
                     std::shared_ptr<einsums::GeneralRuntimeTensor<T, std::allocator<T>>>, einsums::tensor_base::RuntimeTensorNoType>;

template <typename T>
using ExportTensorViewClass =
    pybind11::class_<RuntimeTensorView<T>, PyTensorView<T>, SharedRuntimeTensorView<T>, einsums::tensor_base::RuntimeTensorNoType>;

} // namespace detail

void EINSUMS_EXPORT export_tensor_initf(detail::ExportTensorClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_initd(detail::ExportTensorClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_initc(detail::ExportTensorClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_initz(detail::ExportTensorClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_opsf(detail::ExportTensorClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_opsd(detail::ExportTensorClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_opsc(detail::ExportTensorClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_opsz(detail::ExportTensorClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_manipf(detail::ExportTensorClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_manipd(detail::ExportTensorClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_manipc(detail::ExportTensorClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_manipz(detail::ExportTensorClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_queryf(detail::ExportTensorClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_queryd(detail::ExportTensorClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_queryc(detail::ExportTensorClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_queryz(detail::ExportTensorClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensorf(pybind11::module &mod);
void EINSUMS_EXPORT export_tensord(pybind11::module &mod);
void EINSUMS_EXPORT export_tensorc(pybind11::module &mod);
void EINSUMS_EXPORT export_tensorz(pybind11::module &mod);

void EINSUMS_EXPORT export_tensor_view_initf(detail::ExportTensorViewClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_view_initd(detail::ExportTensorViewClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_view_initc(detail::ExportTensorViewClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_view_initz(detail::ExportTensorViewClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_view_opsf(detail::ExportTensorViewClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_view_opsd(detail::ExportTensorViewClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_view_opsc(detail::ExportTensorViewClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_view_opsz(detail::ExportTensorViewClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_view_manipf(detail::ExportTensorViewClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_view_manipd(detail::ExportTensorViewClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_view_manipc(detail::ExportTensorViewClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_view_manipz(detail::ExportTensorViewClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_view_queryf(detail::ExportTensorViewClass<float> &tensor);
void EINSUMS_EXPORT export_tensor_view_queryd(detail::ExportTensorViewClass<double> &tensor);
void EINSUMS_EXPORT export_tensor_view_queryc(detail::ExportTensorViewClass<std::complex<float>> &tensor);
void EINSUMS_EXPORT export_tensor_view_queryz(detail::ExportTensorViewClass<std::complex<double>> &tensor);

void EINSUMS_EXPORT export_tensor_viewf(pybind11::module &mod);
void EINSUMS_EXPORT export_tensor_viewd(pybind11::module &mod);
void EINSUMS_EXPORT export_tensor_viewc(pybind11::module &mod);
void EINSUMS_EXPORT export_tensor_viewz(pybind11::module &mod);

} // namespace einsums::python