//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>

#include <complex>
#include <memory>

namespace einsums::tensor_algebra {

template <typename IndexContainer>
class EINSUMS_EXPORT AbstractContraction {
  public:
    virtual ~AbstractContraction() = default;

    virtual void execute(IndexContainer &C_index, IndexContainer &A_index, IndexContainer &B_index) = 0;
};

template <bool ConjA, bool ConjB, FunctionTensorConcept CType, FunctionTensorConcept AType, FunctionTensorConcept BType,
          typename ABDataType, typename IndexContainer>
class Contraction : public AbstractContraction<IndexContainer> {
  public:
    constexpr static ptrdiff_t NO_INDEX = -1;

    Contraction(Contraction<ConjA, ConjB, CType, AType, BType, ABDataType, IndexContainer> *previous, ptrdiff_t C_index, ptrdiff_t A_index,
                ptrdiff_t B_index)
        : A_index_{A_index}, B_index_{B_index}, C_index_{C_index} {
        previous->next_ = this;
    }

    virtual ~Contraction() {
        if (!is_leaf()) {
            delete next_;
            next_ = nullptr;
        }
    }

    Contraction<ConjA, ConjB, CType, AType, BType, ABDataType, IndexContainer> *next() { return next_; }

    [[nodiscard]] bool is_leaf() const { return (next_ == nullptr); }

    void attach_C(CType *C) {
        C_ = C;
        if (!is_leaf()) {
            next_->attach_C(C);
        }
    }

    void attach_A(AType const *A) {
        A_ = A;
        if (!is_leaf()) {
            next_->attach_A(A);
        }
    }

    void attach_B(BType const *B) {
        B_ = B;
        if (!is_leaf()) {
            next_->attach_B(B);
        }
    }

    void set_AB_prefactor(ABDataType prefactor) {
        AB_prefactor_ = prefactor;
        if (!is_leaf()) {
            next_->set_AB_prefactor(prefactor);
        }
    }

    void update_dims() {
        if (A_index_ != NO_INDEX) {
            dim_ = A_->dim(A_index_);
        } else if (B_index_ != NO_INDEX) {
            dim_ = B_->dim(B_index_);
        } else if (C_index_ != NO_INDEX) {
            dim_ = C_->dim(C_index_);
        }

        if (!is_leaf()) {
            next_->update_dims();
        }
    }

    void execute(IndexContainer &C_indices, IndexContainer &A_indices, IndexContainer &B_indices) override {
        if (!is_leaf()) {
            for (size_t i = 0; i < dim_; i++) {
                if (A_index_ >= 0) {
                    A_indices[A_index_] = i;
                }

                if (B_index_ >= 0) {
                    B_indices[B_index_] = i;
                }

                if (C_index_ >= 0) {
                    C_indices[C_index_] = i;
                }

                next_->execute(C_indices, A_indices, B_indices);
            }
        } else {
            // Select based on which elements are not modified.
            if (A_index_ == NO_INDEX && B_index_ == NO_INDEX && C_index_ == NO_INDEX) {
                EINSUMS_THROW_EXCEPTION(uninitialized_error, "Can not perform a contraction where no tensor has indices!");
            } else if (A_index_ == NO_INDEX && B_index_ == NO_INDEX && C_index_ != NO_INDEX) {
                auto A_val = (*A_)(A_indices);
                auto B_val = (*B_)(B_indices);

                if constexpr (ConjA && IsComplexTensor<AType>) {
                    A_val = std::conj(A_val);
                }

                if constexpr (ConjB && IsComplexTensor<BType>) {
                    B_val = std::conj(B_val);
                }

                auto const product = A_val * B_val * AB_prefactor_;

                for (size_t i = 0; i < dim_; i++) {
                    C_indices[C_index_] = i;

                    (*C_)(C_indices) += product;
                }
            } else if (A_index_ == NO_INDEX && B_index_ != NO_INDEX && C_index_ == NO_INDEX) {
                auto A_val = (*A_)(A_indices);

                if constexpr (ConjA && IsComplexTensor<AType>) {
                    A_val = std::conj(A_val);
                }

                auto const        product = A_val * AB_prefactor_;
                ValueTypeT<CType> sum{0.0};

                for (size_t i = 0; i < dim_; i++) {
                    B_indices[B_index_] = i;

                    auto B_val = (*B_)(B_indices);

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    sum += product * B_val;
                }

                (*C_)(C_indices) += sum;
            } else if (A_index_ == NO_INDEX && B_index_ != NO_INDEX && C_index_ != NO_INDEX) {
                auto A_val = (*A_)(A_indices);

                if constexpr (ConjA && IsComplexTensor<AType>) {
                    A_val = std::conj(A_val);
                }

                auto const product = A_val * AB_prefactor_;

                for (size_t i = 0; i < dim_; i++) {
                    B_indices[B_index_] = i;
                    C_indices[C_index_] = i;

                    auto B_val = (*B_)(B_indices);

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    (*C_)(C_indices) += product * B_val;
                }
            } else if (A_index_ != NO_INDEX && B_index_ == NO_INDEX && C_index_ == NO_INDEX) {
                auto B_val = (*B_)(B_indices);

                if constexpr (ConjB && IsComplexTensor<BType>) {
                    B_val = std::conj(B_val);
                }

                auto const        product = B_val * AB_prefactor_;
                ValueTypeT<CType> sum{0.0};

                for (size_t i = 0; i < dim_; i++) {
                    A_indices[A_index_] = i;

                    auto A_val = (*A_)(A_indices);

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    sum += product * A_val;
                }

                (*C_)(C_indices) += sum;
            } else if (A_index_ != NO_INDEX && B_index_ == NO_INDEX && C_index_ != NO_INDEX) {
                auto B_val = (*B_)(B_indices);

                if constexpr (ConjB && IsComplexTensor<BType>) {
                    B_val = std::conj(B_val);
                }

                auto const product = B_val * AB_prefactor_;

                for (size_t i = 0; i < dim_; i++) {
                    A_indices[A_index_] = i;
                    C_indices[C_index_] = i;

                    auto A_val = (*A_)(A_indices);

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    (*C_)(C_indices) += product * A_val;
                }
            } else if (A_index_ != NO_INDEX && B_index_ != NO_INDEX && C_index_ == NO_INDEX) {
                ValueTypeT<CType> sum{0.0};

                for (size_t i = 0; i < dim_; i++) {
                    A_indices[A_index_] = i;
                    B_indices[B_index_] = i;

                    auto A_val = (*A_)(A_indices);

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    auto B_val = (*B_)(B_indices);

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    sum += AB_prefactor_ * A_val * B_val;
                }

                (*C_)(C_indices) += sum;
            } else {
                for (size_t i = 0; i < dim_; i++) {
                    A_indices[A_index_] = i;
                    B_indices[B_index_] = i;
                    C_indices[C_index_] = i;

                    auto A_val = (*A_)(A_indices);

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    auto B_val = (*B_)(B_indices);

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    (*C_)(C_indices) = AB_prefactor_ * A_val * B_val;
                }
            }
        }
    }

  protected:
    Contraction<ConjA, ConjB, CType, AType, BType, ABDataType, IndexContainer> *next_{nullptr};

    CType       *C_{nullptr};
    AType const *A_{nullptr};
    BType const *B_{nullptr};
    ABDataType   AB_prefactor_;

    size_t    dim_{0};
    ptrdiff_t A_index_{NO_INDEX}, B_index_{NO_INDEX}, C_index_{NO_INDEX};
};

template <bool ConjA, bool ConjB, typename CType, BasicTensorConcept AType, BasicTensorConcept BType, typename ABDataType,
          typename IndexContainer>
    requires(ScalarConcept<CType> || BasicTensorConcept<CType>)
class BasicTensorContraction {
    BasicTensorContraction(BasicTensorContraction<ConjA, ConjB, CType, AType, BType, ABDataType, IndexContainer> *previous,
                           ptrdiff_t C_index, ptrdiff_t A_index, ptrdiff_t B_index)
        : A_index_{A_index}, B_index_{B_index}, C_index_{Cindex} {
        previous->next_ = this;
    }

    virtual ~BasicTensorContraction() {
        if (!is_leaf()) {
            delete next_;
            next_ = nullptr;
        }
    }

    BasicTensorContraction<ConjA, ConjB, CType, AType, BType, ABDataType, IndexContainer> *next() { return next_; }

    [[nodiscard]] bool is_leaf() const { return (next_ == nullptr); }

    void attach_C(CType *C) {
        C_ = C;
        if (!is_leaf()) {
            next_->attach_C(C);
        }
    }

    void attach_A(AType const *A) {
        A_ = A;
        if (!is_leaf()) {
            next_->attach_A(A);
        }
    }

    void attach_B(BType const *B) {
        B_ = B;
        if (!is_leaf()) {
            next_->attach_B(B);
        }
    }

    void set_AB_prefactor(ABDataType prefactor) {
        AB_prefactor_ = prefactor;
        if (!is_leaf()) {
            next_->set_AB_prefactor(prefactor);
        }
    }

    void update_dims() {
        if (A_index_ != NO_INDEX) {
            dim_ = A_->dim(A_index_);
        } else if (B_index_ != NO_INDEX) {
            dim_ = B_->dim(B_index_);
        } else if (C_index_ != NO_INDEX) {
            if constexpr (TensorConcept<CType>) {
                dim_ = C_->dim(C_index_);
            }
        }

        if (!is_leaf()) {
            next_->update_dims();
        }
    }

    void update_strides() {
        if (A_index_ != NO_INDEX) {
            A_stride_ = A_->stride(A_index_);
        } else {
            A_stride_ = 0;
        }

        if (B_index_ != NO_INDEX) {
            B_stride_ = B_->stride(B_index_);
        } else {
            B_stride_ = 0;
        }

        if constexpr (TensorConcept<CType>) {
            if (C_index_ != NO_INDEX) {
                C_stride_ = C_->stride(C_index_);
            } else {
                C_stride_ = 0;
            }
        } else {
            C_stride_ = 0;
        }

        if (!is_leaf()) {
            next_->update_strides();
        }
    }

    void execute(IndexContainer &C_indices, IndexContainer &A_indices, IndexContainer &B_indices) override {
        if constexpr (TensorConcept<CType>) {
            execute_pass(C_->data(C_indices), A_->data(A_indices), B_->data(B_indices));
        } else {
            execute_pass(C_, A_->data(A_indices), B_->data(B_indices));
        }
    }

  protected:
    void execute_pass(ValueTypeT<CType> *C_ptr, ValueTypeT<AType> const *A_ptr, ValueTypeT<BType> const *B_ptr) {
        if (!is_leaf()) {
            for (size_t i = 0; i < dim_; i++) {
                next_->execute_pass(C_ptr + i * C_stride_, A_ptr + i * A_stride_, B_ptr + i * B_stride_);
            }
        } else {
            // Select based on which elements are not modified.
            if (A_index_ == NO_INDEX && B_index_ == NO_INDEX && C_index_ == NO_INDEX) {
                EINSUMS_THROW_EXCEPTION(uninitialized_error, "Can not perform a contraction where no tensor has indices!");
            } else if (A_index_ == NO_INDEX && B_index_ == NO_INDEX && C_index_ != NO_INDEX) {
                auto A_val = *A_ptr;
                auto B_val = *B_ptr;

                if constexpr (ConjA && IsComplexTensor<AType>) {
                    A_val = std::conj(A_val);
                }

                if constexpr (ConjB && IsComplexTensor<BType>) {
                    B_val = std::conj(B_val);
                }

                auto const product = A_val * B_val * AB_prefactor_;
				size_t C_index = 0;

                for (size_t i = 0; i < dim_; i++, C_index += C_stride_) {
                    C_ptr[C_index] += product;
                }
            } else if (A_index_ == NO_INDEX && B_index_ != NO_INDEX && C_index_ == NO_INDEX) {
                auto A_val = *A_ptr;

                if constexpr (ConjA && IsComplexTensor<AType>) {
                    A_val = std::conj(A_val);
                }

                auto const        product = A_val * AB_prefactor_;
                ValueTypeT<CType> sum{0.0};
				size_t B_index = 0;

                for (size_t i = 0; i < dim_; i++, B_index += B_stride_) {
                    auto B_val = B_ptr[B_index];

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    sum += product * B_val;
                }

                *C_ptr += sum;
            } else if (A_index_ == NO_INDEX && B_index_ != NO_INDEX && C_index_ != NO_INDEX) {
                auto A_val = *A_ptr;

                if constexpr (ConjA && IsComplexTensor<AType>) {
                    A_val = std::conj(A_val);
                }

                auto const product = A_val * AB_prefactor_;
				size_t B_index = 0, C_index = 0;

                for (size_t i = 0; i < dim_; i++, B_index += B_stride_, C_index += C_stride_) {
                    auto B_val = B_ptr[B_index];

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    C_ptr[C_index] += product * B_val;
                }
            } else if (A_index_ != NO_INDEX && B_index_ == NO_INDEX && C_index_ == NO_INDEX) {
                auto B_val = *B_ptr;

                if constexpr (ConjB && IsComplexTensor<BType>) {
                    B_val = std::conj(B_val);
                }

                auto const        product = B_val * AB_prefactor_;
                ValueTypeT<CType> sum{0.0};
				
				size_t A_index = 0;

                for (size_t i = 0; i < dim_; i++, A_index += A_stride_) {
                    auto A_val = A_ptr[A_index];

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    sum += product * A_val;
                }

                *C_ptr += sum;
            } else if (A_index_ != NO_INDEX && B_index_ == NO_INDEX && C_index_ != NO_INDEX) {
                auto B_val = *B_ptr;

                if constexpr (ConjB && IsComplexTensor<BType>) {
                    B_val = std::conj(B_val);
                }

                auto const product = B_val * AB_prefactor_;
				size_t A_index = 0, C_index = 0;

                for (size_t i = 0; i < dim_; i++, A_index += A_stride_, C_index += C_stride_) {
                    auto A_val = A_ptr[A_index];

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    C_ptr[C_index] += product * A_val;
                }
            } else if (A_index_ != NO_INDEX && B_index_ != NO_INDEX && C_index_ == NO_INDEX) {
                ValueTypeT<CType> sum{0.0};
				size_t A_index = 0, B_index = 0;

                for (size_t i = 0; i < dim_; i++, A_index += A_stride_, B_index += B_stride_) {
                    auto A_val = A_ptr[A_index];

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    auto B_val = B_ptr[B_index];

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    sum += AB_prefactor_ * A_val * B_val;
                }

                *C_ptr += sum;
            } else {
				size_t A_index = 0, B_index = 0, C_index = 0;
                for (size_t i = 0; i < dim_; i++, A_index += A_stride_, B_index += B_stride_, C_index += C_stride_) {
                    auto A_val = A_ptr[A_index];

                    if constexpr (ConjA && IsComplexTensor<AType>) {
                        A_val = std::conj(A_val);
                    }

                    auto B_val = B_ptr[B_index];

                    if constexpr (ConjB && IsComplexTensor<BType>) {
                        B_val = std::conj(B_val);
                    }

                    C_ptr[C_index] = AB_prefactor_ * A_val * B_val;
                }
            }
        }
    }

    BasicTensorContraction<ConjA, ConjB, CType, AType, BType, ABDataType, IndexContainer> *next_{nullptr};

    CType       *C_{nullptr};
    AType const *A_{nullptr};
    BType const *B_{nullptr};
    ABDataType   AB_prefactor_;

    size_t    dim_{0}, A_stride_{0}, B_stride_{0}, C_stride_{0};
    ptrdiff_t A_index_{NO_INDEX}, B_index_{NO_INDEX}, C_index_{NO_INDEX};
};

} // namespace einsums::tensor_algebra