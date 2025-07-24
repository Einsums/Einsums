//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Subset RuntimeTensorView", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    SECTION("Subset View 7x7[1,:] -> 1x7") {
        size_t const size = 7;
        size_t const row  = 1;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size);
        auto                    I_view     = I_original(row, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(row, i) == I_view(i));
        }
    }

    SECTION("Subset RuntimeView 7x7x7[4,:,:] -> 7x7") {
        size_t const size = 7;
        size_t const d1   = 4;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size, size);
        auto                    I_view     = I_original(d1, All, All);

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                REQUIRE(I_original(d1, i, j) == I_view(i, j));
            }
        }
    }

    SECTION("Subset RuntimeView 7x7x7[4,3,:] -> 7") {
        size_t const size = 7;
        size_t const d1   = 4;
        size_t const d2   = 3;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size, size);
        auto                    I_view     = I_original(d1, d2, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(d1, d2, i) == I_view(i));
        }
    }
}

TEMPLATE_TEST_CASE("Subset RuntimeTensor Conversion", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    SECTION("Subset View 7x7[1,:] -> 1x7") {
        size_t const size = 7;
        size_t const row  = 1;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size);
        auto                    I_view     = (TensorView<TestType, 1>)I_original(row, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(row, i) == I_view(i));
        }
    }

    SECTION("Subset RuntimeView 7x7x7[4,:,:] -> 7x7") {
        size_t const size = 7;
        size_t const d1   = 4;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size, size);
        auto                    I_view     = (TensorView<TestType, 2>)I_original(d1, All, All);

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                REQUIRE(I_original(d1, i, j) == I_view(i, j));
            }
        }
    }

    SECTION("Subset RuntimeView 7x7x7[4,3,:] -> 7") {
        size_t const size = 7;
        size_t const d1   = 4;
        size_t const d2   = 3;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size, size);
        auto                    I_view     = (TensorView<TestType, 1>)I_original(d1, d2, All);

        for (size_t i = 0; i < size; i++) {
            REQUIRE(I_original(d1, d2, i) == I_view(i));
        }
    }

    SECTION("Full View") {
        size_t const size = 7;

        RuntimeTensor<TestType> I_original = create_random_tensor<TestType>("Original", size, size);
        auto                    I_view     = (TensorView<TestType, 2>)I_original;

        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                REQUIRE(I_original(i, j) == I_view(i, j));
            }
        }
    }
}

TEST_CASE("Runtime Tensor Assignment") {
    using namespace einsums;
    RuntimeTensor<double> A{"A", std::vector<size_t>{10, 10}};
    A = create_random_tensor("A", 10, 10);
    RuntimeTensor<double> C{std::vector<size_t>{10, 10}};
    C                       = create_random_tensor("C", 10, 10);
    RuntimeTensor<double> D = create_random_tensor("D", 20, 20);
    RuntimeTensor<double> E;
    RuntimeTensor<double> F = C;
    E                       = A;
    auto D_view             = D(Range{0, 10}, Range{0, 10});

    REQUIRE(A.rank() == 2);

    Tensor<double, 4> B_base = create_random_tensor("B", 10, 10, 10, 10);

    RuntimeTensor<double> B = (RuntimeTensor<double>)B_base(Range{0, 5}, Range{1, 6}, Range{2, 7}, Range{3, 8});

    REQUIRE(B.rank() == 4);

    REQUIRE(A.data() != nullptr);
    REQUIRE(B.data() != nullptr);
    REQUIRE(C.data() != nullptr);
    REQUIRE(D.data() != nullptr);
    REQUIRE((E.data() != nullptr && E.data() != A.data()));
    REQUIRE((F.data() != nullptr && F.data() != C.data()));

    REQUIRE(A.data(std::array<ptrdiff_t, 2>{-1, 1}) != nullptr);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    REQUIRE(B(std::vector<ptrdiff_t>{i, j, k, l}) == B_base(i, j + 1, k + 2, l + 3));
                }
            }
        }
    }

    A = C;

    auto const &C_const = *&C;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(A(i, j) == C_const(i, j));
        }
    }

    A                   = D_view;
    auto const &D_const = *&D;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(A(std::array<int, 2>{i, j}) == D_const(std::array<int, 2>{i, j}));
        }
    }

    D_view.zero();

    D_view = 1.0;

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            if (i < 10 && j < 10) {
                REQUIRE(D(i, j) == 1.0);
            }
        }
    }

    // initializer list constructors
    REQUIRE_NOTHROW(RuntimeTensor<double>("test_tensor", {}));
    REQUIRE_NOTHROW(RuntimeTensor<double>("test_tensor", {3, 4, 5}));
    REQUIRE_NOTHROW(RuntimeTensor<double>({3, 4, 5}));
}

TEST_CASE("Runtime Tensor View Creation") {
    using namespace einsums;

    RuntimeTensor<double>        Base       = create_random_tensor("Base", 10, 10, 10);
    RuntimeTensor<double> const &const_base = Base;

    RuntimeTensorView<double>        base_view{Base};
    RuntimeTensorView<double> const &const_base_view = base_view;

    Tensor<double, 3>        rank_base       = create_random_tensor("rank_base", 10, 10, 10);
    Tensor<double, 3> const &const_rank_base = rank_base;

    TensorView<double, 3>        rank_view       = rank_base(All, All, All);
    TensorView<double, 3> const &const_rank_view = rank_view;

    RuntimeTensorView<double> A{Base, std::vector<size_t>{10, 100}}, B{A, std::vector<size_t>{100, 10}},
        C{Base, std::vector<size_t>{5, 5, 5}, std::vector<size_t>{100, 10, 1}, std::vector<size_t>{1, 2, 3}},
        D{RuntimeTensorView<double>(Base), std::vector<size_t>{5, 5, 5}, std::vector<size_t>{100, 10, 1}, std::vector<size_t>{1, 2, 3}},
        E{rank_view}, F{rank_base};

    RuntimeTensorView<double> const G{const_base, std::vector<size_t>{10, 100}}, H{const_base_view, std::vector<size_t>{100, 10}},
        I{const_rank_view}, J{const_rank_base};

    RuntimeTensorView<double> K = A(All, Range{0, 10});

    RuntimeTensorView<double> const L = G(All, Range{0, 10});

    REQUIRE(A.rank() == 2);
    REQUIRE(B.rank() == 2);
    REQUIRE(C.rank() == 3);
    REQUIRE(D.rank() == 3);
    REQUIRE(E.rank() == 3);
    REQUIRE(F.rank() == 3);
    REQUIRE(G.rank() == 2);
    REQUIRE(H.rank() == 2);
    REQUIRE(I.rank() == 3);
    REQUIRE(J.rank() == 3);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                REQUIRE(A(i, j * 10 + k) == Base(i, j, k));
                REQUIRE(B(i * 10 + j, k) == Base(i, j, k));
                REQUIRE(E(i, j, k) == rank_base(i, j, k));
                REQUIRE(F(i, j, k) == rank_base(i, j, k));
                REQUIRE(G(i, j * 10 + k) == Base(i, j, k));
                REQUIRE(H(i * 10 + j, k) == Base(i, j, k));
                REQUIRE(I(i, j, k) == rank_base(i, j, k));
                REQUIRE(J(i, j, k) == rank_base(i, j, k));
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                REQUIRE(C(i, j, k) == Base(i + 1, j + 2, k + 3));
                REQUIRE(D(i, j, k) == Base(i + 1, j + 2, k + 3));
            }
        }
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(K(i, j) == A(i, j));
            REQUIRE(L(i, j) == G(i, j));
        }
    }
}

TEST_CASE("Runtime Tensor View Assignment") {
    using namespace einsums;

    RuntimeTensor<double> Base      = create_random_tensor("Base", 10, 10, 10);
    RuntimeTensor<double> Base_copy = Base;

    RuntimeTensorView<double>        A       = Base(Range{5, 10}, Range{5, 10}, Range{5, 10});
    RuntimeTensorView<double> const &const_A = A;

    RuntimeTensor<double>        B       = create_random_tensor("B", 5, 5, 5);
    RuntimeTensor<double> const &const_B = B;

    RuntimeTensor<double>            Base2   = create_random_tensor("Base2", 10, 10, 10);
    RuntimeTensorView<double>        C       = Base2(Range{0, 5}, Range{0, 5}, Range{0, 5});
    RuntimeTensorView<double> const &const_C = C;

    Tensor<double, 3>     D = create_random_tensor("D", 5, 5, 5);
    TensorView<double, 3> E = D(All, All, All);

    Tensor<double, 3> const    &const_D = D;
    TensorView<double, 3> const const_E = E;

    A.zero();

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == 0);
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = 1.0;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == 1);
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = B;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == B(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = C;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == C(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_B;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == B(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_C;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == C(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = D;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == D(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = E;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == E(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_D;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == D(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_E;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                if (i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == E(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }
}