//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/HPTT/Transpose.hpp>

#include <memory>

#include "Einsums/LinearAlgebra.hpp"

namespace einsums::tensor_algebra::detail {

/**
 * @class Command
 *
 * Virtual class for defining commands.
 */
class Command {
    // Implementation taken from the Gang of Four, page 239
  public:
    virtual ~Command() = default;

    virtual void execute() = 0;

  protected:
    Command() = default;
};

/**
 * @class PermuteCommand
 *
 * Command for performing a permutation.
 */
template <typename T>
class PermuteCommand : public Command {
  public:
    PermuteCommand(std::shared_ptr<hptt::Transpose<T>> &plan) : _transpose_plan{plan} {}

    ~PermuteCommand() = default;

    std::shared_ptr<hptt::Transpose<T>> get_plan() { return _transpose_plan; }

    void execute() override { _transpose_plan->execute(); }

  private:
    std::shared_ptr<hptt::Transpose<T>> _transpose_plan;
};

template <typename CType, TensorConcept AType, TensorConcept BType>
class AbstractLoopReceiver {
  public:
    virtual ~AbstractLoopReceiver() = default;

    virtual void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                             BType const &B) const = 0;
};

template <TensorConcept CType, TensorConcept AType, TensorConcept BType>
class CoreLoopReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    CoreLoopReceiver(size_t stride_C, size_t stride_A, size_t stride_B, size_t dim, bool conjA, bool conjB)
        : _stride_A{stride_A}, _stride_B{stride_B}, _stride_C{stride_C}, _dim{dim}, _conjA{conjA}, _conjB{conjB} {}

    ~CoreLoopReceiver() = default;

    void add_child(std::shared_ptr<AbstractLoopReceiver<CType, AType, BType>> const &next) { _next = next; }

    void remove_child() { _next.reset(); }

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        // Save the variables, possibly in registers to avoid having to constantly dereference the loop object.
        size_t const stride_A = _stride_A, stride_B = _stride_B, stride_C = _stride_C;
        CType       *C_ptr = C->data();
        AType const *A_ptr = A->data();
        BType const *B_ptr = B->data();
        if (_next == nullptr) {
            // This is a leaf node if next is nullptr.
            if constexpr (IsComplexV<AType> || IsComplexV<BType>) {
                if (_conjA && _conjB) {
                    for (size_t i = 0; i < _dim; i++) {
                        *C_ptr += AB_prefactor * std::conj(*A_ptr) * std::conj(*B_ptr);

                        A_ptr += stride_A;
                        B_ptr += stride_B;
                        C_ptr += stride_C;
                    }
                } else if (_conjA) {
                    for (size_t i = 0; i < _dim; i++) {
                        *C_ptr += AB_prefactor * std::conj(*A_ptr) * *B_ptr;

                        A_ptr += stride_A;
                        B_ptr += stride_B;
                        C_ptr += stride_C;
                    }
                } else if (_conjB) {
                    for (size_t i = 0; i < _dim; i++) {
                        *C_ptr += AB_prefactor * *A_ptr * std::conj(*B_ptr);

                        A_ptr += stride_A;
                        B_ptr += stride_B;
                        C_ptr += stride_C;
                    }
                } else {
                    for (size_t i = 0; i < _dim; i++) {
                        *C_ptr += AB_prefactor * *A_ptr * *B_ptr;

                        A_ptr += stride_A;
                        B_ptr += stride_B;
                        C_ptr += stride_C;
                    }
                }
            } else {
                for (size_t i = 0; i < _dim; i++) {
                    *C_ptr += AB_prefactor * *A_ptr * *B_ptr;

                    A_ptr += stride_A;
                    B_ptr += stride_B;
                    C_ptr += stride_C;
                }
            }
        } else {
            // This is not a leaf node. Call the child.
            for (size_t i = 0; i < _dim; i++) {
                _next->loop_action(C, AB_prefactor, A, B);
            }
        }
    }

  protected:
    size_t _stride_A, _stride_B, _stride_C;
    size_t _dim;

    std::shared_ptr<AbstractLoopReceiver<CType, AType, BType>> _next{nullptr};

    bool _conjA, _conjB;
};

template <typename CType, TensorConcept AType, TensorConcept BType>
class DotReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    DotReceiver(bool conjA, bool conjB) : _conjA{conjA}, _conjB{conjB} {}

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        if constexpr (IsComplexV<AType> || IsComplexV<BType>) {
            if (_conjA && _conjB) {
                *C += AB_prefactor * std::conj(einsums::linear_algebra::dot(A, B));
            } else if (_conjA) {
                *C += AB_prefactor * std::conj(einsums::linear_algebra::true_dot(A, B));
            } else if (_conjB) {
                *C += AB_prefactor * einsums::linear_algebra::true_dot(A, B);
            } else {
                *C += AB_prefactor * einsums::linear_algebra::dot(A, B);
            }
        } else {
            *C += AB_prefactor * einsums::linear_algebra::dot(A, B);
        }
    }

  private:
    bool _conjA, _conjB;
};

template <TensorConcept CType, TensorConcept AType, TensorConcept BType>
class GemmReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    GemmReceiver(char transA, char transB) : _transA{transA}, _transB{transB} {}

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        einsums::linear_algebra::gemm(_transA, _transB, AB_prefactor, A, B, ValueTypeT<CType>{1.0}, C);
    }

  private:
    char _transA, _transB;
};

template <TensorConcept CType, TensorConcept AType, TensorConcept BType>
class GemvReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    GemvReceiver(char transA, bool swap) : _transA{transA}, _swap{swap} {}

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        if (!_swap) {
            einsums::linear_algebra::gemv(_transA, AB_prefactor, A, B, ValueTypeT<CType>{1.0}, C);
        } else {
            einsums::linear_algebra::gemv(_transA, AB_prefactor, B, A, ValueTypeT<CType>{1.0}, C);
        }
    }

  private:
    char _transA;
    bool _swap;
};

template <TensorConcept CType, TensorConcept AType, TensorConcept BType>
class GerReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    GerReceiver() = default;

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        einsums::linear_algebra::ger(AB_prefactor, A, B, C);
    }
};

template <TensorConcept CType, TensorConcept AType, TensorConcept BType>
class GercReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    GercReceiver() = default;

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        einsums::linear_algebra::gerc(AB_prefactor, A, B, C);
    }
};

template <TensorConcept CType, TensorConcept AType, TensorConcept BType>
class DirProdReceiver : public AbstractLoopReceiver<CType, AType, BType> {
  public:
    DirProdReceiver() = default;

    void loop_action(CType *C, BiggestTypeT<ValueTypeT<AType>, ValueTypeT<BType>> AB_prefactor, AType const &A,
                     BType const &B) const override {
        einsums::linear_algebra::direct_product(AB_prefactor, A, B, ValueTypeT<CType>{1.0}, C);
    }
};

template <typename CType>
class ScaleCommand : public Command {
  public:
    ScaleCommand() = default;

  protected:
    CType *_C{nullptr};

    ValueTypeT<CType> _C_prefactor{0.0};
};

template <typename CType, TensorConcept AType, TensorConcept BType>
class EinsumCommand : public Command {
  public:
    EinsumCommand() = default;

    ~EinsumCommand() = default;

    void set_data(CType *C, BiggestTypeT<typename AType::ValueType, typename BType::ValueType> AB_prefactor, AType const &A,
                  BType const &B) {
        _C            = C;
        _A            = &A;
        _B            = &B;
        _AB_prefactor = AB_prefactor;
    }

    void set_receiver(std::shared_ptr<AbstractLoopReceiver<CType, AType, BType>> receiver) { _receiver = receiver; }

    void execute() override { _receiver->loop_action(_C, _AB_prefactor, *_A, *_B); }

  protected:
    CType *_C{nullptr};

    AType const *_A{nullptr};
    BType const *_B{nullptr};

    BiggestTypeT<typename AType::ValueType, typename BType::ValueType> _AB_prefactor{1.0};

    std::shared_ptr<AbstractLoopReceiver<CType, AType, BType>> _receiver;
};

} // namespace einsums::tensor_algebra::detail