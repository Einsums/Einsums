//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include "einsums/TensorAlgebra.hpp"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"

#include <H5Fpublic.h>
#include <catch2/catch_all.hpp>
#include <complex>



TEST_CASE("einsum_gemv") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    SECTION("check") {
        size_t _p = 7, _q = 7, _r = 7, _s = 7;

        Tensor g = create_random_tensor("g", _p, _q, _r, _s);
        Tensor D = create_random_tensor("d", _r, _s);

        Tensor F{"F", _p, _q};
        Tensor F0{"F0", _p, _q};

        zero(F);
        zero(F0);

        REQUIRE_NOTHROW(einsum(1.0, Indices{p, q}, &F0, 2.0, Indices{p, q, r, s}, g, Indices{r, s}, D));

        TensorView gv{g, Dim<2>{_p * _q, _r * _s}};
        TensorView dv{D, Dim<1>{_r * _s}};
        TensorView Fv{F, Dim<1>{_p * _q}};

        linear_algebra::gemv<false>(2.0, gv, dv, 1.0, &Fv);

        // println(F0);
        // println(F);
    }
}



TEST_CASE("F12 - V term") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{1}, ncabs{4}, nobs{2};
    int nall{nobs + ncabs};

    auto F = create_incremented_tensor("F", nall, nall, nall, nall);
    auto G = create_incremented_tensor("G", nall, nall, nall, nall);

    TensorView F_ooco{F, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    TensorView F_oooc{F, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    TensorView F_oopq{F, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};
    TensorView G_ooco{G, Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    TensorView G_oooc{G, Dim<4>{nocc, nocc, nocc, ncabs}, Offset<4>{0, 0, 0, nobs}};
    TensorView G_oopq{G, Dim<4>{nocc, nocc, nobs, nobs}, Offset<4>{0, 0, 0, 0}};

    Tensor ijkl_1 = Tensor{"Einsum Temp 1", nocc, nocc, nocc, nocc};
    Tensor ijkl_2 = Tensor{"Einsum Temp 2", nocc, nocc, nocc, nocc};
    Tensor ijkl_3 = Tensor{"Einsum Temp 3", nocc, nocc, nocc, nocc};

    ijkl_1.set_all(0.0);
    ijkl_2.set_all(0.0);
    ijkl_3.set_all(0.0);

    Tensor result  = Tensor{"Result", nocc, nocc, nocc, nocc};
    Tensor result2 = Tensor{"Result2", nocc, nocc, nocc, nocc};

    // println(F);
    // println(G);

    einsum(Indices{i, j, k, l}, &ijkl_1, Indices{i, j, p, n}, G_ooco, Indices{k, l, p, n}, F_ooco);
    einsum(Indices{i, j, k, l}, &ijkl_2, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc);
    einsum(Indices{i, j, k, l}, &ijkl_3, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq);

    result.set_all(0.0);
    result2.set_all(0.0);
    timer::push("raw for loops");
    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    for (size_t _p = 0; _p < ncabs; _p++) {
                        for (size_t _n = 0; _n < nocc; _n++) {
                            // println("A({}, {}, {}, {}) = {}", _i, _j, _p, _n, G_ooco(_i, _j, _p, _n));
                            // println("B({}, {}, {}, {}) = {}", _k, _l, _p, _n, F_ooco(_k, _l, _p, _n));

                            result(_i, _j, _k, _l) += G(_i, _j, nobs + _p, _n) * F(_k, _l, nobs + _p, _n);
                            result2(_i, _j, _k, _l) += G_ooco(_i, _j, _p, _n) * F_ooco(_k, _l, _p, _n);
                        }
                    }
                }
            }
        }
    }
    timer::pop();

    // println(result);
    // println(ijkl_1);

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(result2(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(ijkl_1(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }
}

TEST_CASE("B_tilde") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{5}, ncabs{10}, nobs{10};
    assert(nobs > nocc); // sanity check
    int nall{nobs + ncabs}, nvir{nobs - nocc};

    Tensor CD{"CD", nocc, nocc, nvir, nvir};
    Tensor CD0{"CD0", nocc, nocc, nvir, nvir};
    zero(CD);
    zero(CD0);
    auto C    = create_random_tensor("C", nocc, nocc, nvir, nvir);
    auto D    = create_random_tensor("D", nocc, nocc, nvir, nvir);
    auto D_ij = D(2, 2, All, All);

    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, C, Indices{a, b}, D_ij);

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    CD0(_k, _l, _a, _b) = C(_k, _l, _a, _b) * D(2, 2, _a, _b);
                }
            }
        }
    }

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    REQUIRE_THAT(CD(_k, _l, _a, _b), Catch::Matchers::WithinAbs(CD0(_k, _l, _a, _b), 0.000001));
                }
            }
        }
    }
}

TEST_CASE("Khatri-Rao") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    const int _I{8}, _M{4}, _r{16};

    SECTION("einsum") {

        auto KR  = Tensor{"KR", _I, _M, _r};
        auto KR0 = Tensor{"KR0", _I, _M, _r};

        auto T = create_random_tensor("T", _I, _r);
        auto U = create_random_tensor("U", _M, _r);

        einsum(Indices{I, M, r}, &KR, Indices{I, r}, T, Indices{M, r}, U);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    REQUIRE_THAT(KR(x, y, z), Catch::Matchers::WithinAbs(KR0(x, y, z), 0.000001));
                }
            }
        }
    }

    SECTION("special function") {
        auto KR0 = Tensor{"KR0", _I, _M, _r};

        auto T = create_random_tensor("T", _I, _r);
        auto U = create_random_tensor("U", _M, _r);

        auto KR = khatri_rao(Indices{I, r}, T, Indices{M, r}, U);
        // println(result);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        auto KR0_view = TensorView{KR0, Dim<2>{_I * _M, _r}};

        for (int x = 0; x < _I * _M; x++) {
            for (int z = 0; z < _r; z++) {
                REQUIRE_THAT(KR(x, z), Catch::Matchers::WithinAbs(KR0_view(x, z), 0.000001));
            }
        }
    }
}

template <typename TC, typename TA, typename TB>
void einsum_mixed_test() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    const auto i_ = 10, j_ = 10, k_ = 10;

    auto A  = create_random_tensor<TA>("A", i_, k_);
    auto B  = create_random_tensor<TB>("B", k_, j_);
    auto C  = create_tensor<TC>("C", i_, j_);
    auto C0 = create_tensor<TC>("C0", i_, j_);
    zero(C);
    zero(C0);

    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t i = 0; i < i_; i++) {
        for (size_t j = 0; j < j_; j++) {
            for (size_t k = 0; k < k_; k++) {
                C0(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    for (size_t i = 0; i < i_; i++) {
        for (size_t j = 0; j < j_; j++) {
            // println("{:20.14f} {:20.14f} {:20.14f}", C(i, j), C0(i, j), std::abs(C(i, j) - C0(i, j)));
            CHECK(std::abs(C(i, j) - C0(i, j)) < RemoveComplexT<TC>{1.0E-4});
            // REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(C0(i, j), RemoveComplexT<TC>{1.0E-16}));
        }
    }
}

TEST_CASE("einsum-mixed") {
    SECTION("d-f-d") {
        einsum_mixed_test<double, float, double>();
    }
    SECTION("d-f-f") {
        einsum_mixed_test<double, float, float>();
    }
    SECTION("f-d-d") {
        einsum_mixed_test<float, double, double>();
    }
    SECTION("cd-cd-d") {
        einsum_mixed_test<std::complex<double>, std::complex<double>, double>();
    }
    SECTION("d-d-d") {
        einsum_mixed_test<double, double, double>();
    }
    // VERY SENSITIVE
    // SECTION("cf-cd-f") {
    //     einsum_mixed_test<std::complex<float>, std::complex<float>, std::complex<float>>();
    // }
}

template <typename T>
void dot_test() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;
    using namespace einsums::linear_algebra;

    size_t i_{10}, j_{10}, a_{10}, b_{10};

    SECTION("1") {
        auto         A = create_random_tensor<T>("A", i_);
        auto         B = create_random_tensor<T>("B", i_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i}, A, Indices{i}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("2") {
        auto         A = create_random_tensor<T>("A", i_, j_);
        auto         B = create_random_tensor<T>("B", i_, j_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j}, A, Indices{i, j}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("3") {
        auto         A = create_random_tensor<T>("A", i_, j_, a_);
        auto         B = create_random_tensor<T>("B", i_, j_, a_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a}, A, Indices{i, j, a}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }

    SECTION("4") {
        auto         A = create_random_tensor<T>("A", i_, j_, a_, b_);
        auto         B = create_random_tensor<T>("B", i_, j_, a_, b_);
        Tensor<T, 0> C_obtained("C obtained");

        auto C_expected = dot(A, B);

        einsum(Indices{}, &C_obtained, Indices{i, j, a, b}, A, Indices{i, j, a, b}, B);

        if constexpr (!einsums::IsComplexV<T>) {
            REQUIRE_THAT(C_obtained, Catch::Matchers::WithinAbsMatcher(C_expected, 0.0001));
        } else {
            REQUIRE_THAT(((T)C_obtained).real(), Catch::Matchers::WithinAbsMatcher(C_expected.real(), 0.0001));
            REQUIRE_THAT(((T)C_obtained).imag(), Catch::Matchers::WithinAbsMatcher(C_expected.imag(), 0.0001));
        }
    }
}

TEST_CASE("dot") {
    SECTION("float") {
        dot_test<float>();
    }
    SECTION("double") {
        dot_test<double>();
    }
    SECTION("cfloat") {
        dot_test<std::complex<float>>();
    }
    SECTION("cdouble") {
        dot_test<std::complex<double>>();
    }
}

TEST_CASE("Dot TensorView and Tensor") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    const auto i_ = 10, j_ = 10, k_ = 10, l_ = 2;

    auto A  = create_random_tensor<double>("A", i_, k_);
    auto B  = create_random_tensor<double>("B", k_, j_);
    auto C  = create_tensor<double>("C", l_, j_);
    auto C0 = create_tensor<double>("C0", l_, j_);
    zero(C0);

    auto A_view = A(Range{0, l_}, All); // (l_, k_)

    einsum(Indices{l, j}, &C, Indices{l, k}, A_view, Indices{k, j}, B);

    for (size_t l = 0; l < l_; l++) {
        for (size_t j = 0; j < j_; j++) {
            for (size_t k = 0; k < k_; k++) {
                C0(l, j) += A(l, k) * B(k, j);
            }
        }
    }

    for (size_t l = 0; l < l_; l++) {
        for (size_t j = 0; j < j_; j++) {
            // println("{:20.14f} {:20.14f} {:20.14f}", C(l, j), C0(l, j), std::abs(C(l, j) - C0(l, j)));
            REQUIRE_THAT(C(l, j), Catch::Matchers::WithinAbs(C0(l, j), 1e-12));
        }
    }
}

TEST_CASE("andy") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;

    size_t proj_rank_{10}, nocc_{5}, nvirt_{28}, naux_{3}, u_rank_{4};

    SECTION("1") {
        auto y_iW_        = create_random_tensor("y_iW", nocc_, proj_rank_);
        auto y_aW_        = create_random_tensor("y_aW", nvirt_, proj_rank_);
        auto ortho_temp_1 = create_tensor("ortho temp 1", nocc_, nvirt_, proj_rank_);

        zero(ortho_temp_1);
        einsum(0.0, Indices{index::i, index::a, index::W}, &ortho_temp_1, 1.0, Indices{index::i, index::W}, y_iW_,
               Indices{index::a, index::W}, y_aW_);
    }

    SECTION("2") {
        auto tau_         = create_random_tensor("tau", proj_rank_, proj_rank_);
        auto ortho_temp_1 = create_random_tensor("ortho temp 1", nocc_, nvirt_, proj_rank_);
        auto ortho_temp_2 = create_tensor("ortho temp 2", nocc_, nvirt_, proj_rank_);

        zero(ortho_temp_2);
        einsum(0.0, Indices{index::i, index::a, index::P}, &ortho_temp_2, 1.0, Indices{index::i, index::a, index::W}, ortho_temp_1,
               Indices{index::P, index::W}, tau_);
    }

    SECTION("3") {
        auto a = create_random_tensor("a", nvirt_, nvirt_);
        auto b = create_random_tensor("b", nvirt_, nvirt_);
        auto c = create_tensor("c", nvirt_, nvirt_);
        zero(c);

        einsum(0.0, Indices{index::p, index::q}, &c, -1.0, Indices{index::p, index::q}, a, Indices{index::p, index::q}, b);

        for (int x = 0; x < nvirt_; x++) {
            for (int y = 0; y < nvirt_; y++) {
                REQUIRE_THAT(c(x, y), Catch::Matchers::WithinRel(-a(x, y) * b(x, y)));
            }
        }
    }

    SECTION("4") {
        auto A  = create_random_tensor("a", proj_rank_, nocc_, nvirt_);
        auto B  = create_random_tensor("b", nocc_, nvirt_, proj_rank_);
        auto c  = create_tensor("c", proj_rank_, proj_rank_);
        auto c0 = create_tensor("c0", proj_rank_, proj_rank_);

        zero(c);
        einsum(Indices{index::Q, index::X}, &c, Indices{index::Q, index::i, index::a}, A, Indices{index::i, index::a, index::X}, B);

        zero(c0);
        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t X = 0; X < proj_rank_; X++) {
                for (size_t i = 0; i < nocc_; i++) {
                    for (size_t a = 0; a < nvirt_; a++) {
                        c0(Q, X) += A(Q, i, a) * B(i, a, X);
                    }
                }
            }
        }

        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t X = 0; X < proj_rank_; X++) {
                REQUIRE_THAT(c(Q, X), Catch::Matchers::WithinRel(c0(Q, X), 0.00001));
            }
        }
    }

    SECTION("5") {
        auto F_TEMP = create_random_tensor("F_TEMP", proj_rank_, proj_rank_, proj_rank_);
        auto y_aW   = create_random_tensor("y_aW", nvirt_, proj_rank_);
        auto F_BAR  = create_tensor("F_BAR", proj_rank_, nvirt_, proj_rank_);
        auto F_BAR0 = create_tensor("F_BAR", proj_rank_, nvirt_, proj_rank_);

        zero(F_BAR);
        einsum(Indices{index::Q, index::a, index::X}, &F_BAR, Indices{index::Q, index::Y, index::X}, F_TEMP, Indices{index::a, index::Y},
               y_aW);

        zero(F_BAR0);
        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t a = 0; a < nvirt_; a++) {
                for (size_t X = 0; X < proj_rank_; X++) {
                    for (size_t Y = 0; Y < proj_rank_; Y++) {
                        F_BAR0(Q, a, X) += F_TEMP(Q, Y, X) * y_aW(a, Y);
                    }
                }
            }
        }

        for (size_t Q = 0; Q < proj_rank_; Q++) {
            for (size_t a = 0; a < nvirt_; a++) {
                for (size_t X = 0; X < proj_rank_; X++) {
                    REQUIRE_THAT(F_BAR(Q, a, X), Catch::Matchers::WithinRel(F_BAR0(Q, a, X), 0.00001));
                }
            }
        }
    }

    SECTION("6") {
        auto A = create_random_tensor("A", 84);
        auto C = create_tensor("C", 84, 84);
        zero(C);

        einsum(Indices{index::a, index::b}, &C, Indices{index::a}, A, Indices{index::b}, A);

        for (size_t a = 0; a < 84; a++) {
            for (size_t b = 0; b < 84; b++) {
                REQUIRE_THAT(C(a, b), Catch::Matchers::WithinRel(A(a) * A(b), 0.00001));
            }
        }
    }

    SECTION("7") {
        auto A = create_tensor("A", 9);
        A(0)   = 0.26052754;
        A(1)   = 0.20708203;
        A(2)   = 0.18034861;
        A(3)   = 0.18034861;
        A(4)   = 0.10959806;
        A(5)   = 0.10285149;
        A(6)   = 0.10285149;
        A(7)   = 0.10164104;
        A(8)   = 0.06130642;
        auto C = create_tensor("C", 9, 9);
        zero(C);

        einsum(Indices{index::a, index::b}, &C, Indices{index::a}, A, Indices{index::b}, A);

        for (size_t a = 0; a < 9; a++) {
            for (size_t b = 0; b < 9; b++) {
                REQUIRE_THAT(C(a, b), Catch::Matchers::WithinRel(A(a) * A(b), 0.00001));
            }
        }
    }

    SECTION("8") {
        auto C_TILDE = create_random_tensor("C_TILDE", naux_, nvirt_, u_rank_);
        auto B_QY    = create_random_tensor("B_QY", naux_, u_rank_);

        auto D_TILDE = create_tensor("D_TILDE", nvirt_, u_rank_);
        zero(D_TILDE);

        einsum(0.0, Indices{index::a, index::X}, &D_TILDE, 1.0, Indices{index::Q, index::a, index::X}, C_TILDE, Indices{index::Q, index::X},
               B_QY);
    }

    SECTION("9") {
        auto Qov  = create_random_tensor("Qov", naux_, nocc_, nvirt_);
        auto ia_X = create_random_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto N_QX = create_tensor("N_QX", naux_, u_rank_);
        zero(N_QX);

        einsum(Indices{index::Q, index::X}, &N_QX, Indices{index::Q, index::i, index::a}, Qov, Indices{index::i, index::a, index::X}, ia_X);
    }

    SECTION("10") {
        auto t_ia = create_random_tensor("t_ia", nocc_, nvirt_);
        auto ia_X = create_random_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto M_X = create_tensor("M_X", u_rank_);
        zero(M_X);

        einsum(Indices{index::X}, &M_X, Indices{index::i, index::a, index::X}, ia_X, Indices{index::i, index::a}, t_ia);
    }

    SECTION("11") {
        auto B_Qmo = create_random_tensor("Q", naux_, nocc_ + nvirt_, nocc_ + nvirt_);
        // println(B_Qmo);
        auto Qov = B_Qmo(All, Range{0, nocc_}, Range{nocc_, nocc_ + nvirt_});

        // println(Qov, {.full_output = false});

        auto ia_X = create_random_tensor("ia_X", nocc_, nvirt_, u_rank_);

        auto N_QX = create_tensor("N_QX", naux_, u_rank_);
        zero(N_QX);

        einsum(Indices{index::Q, index::X}, &N_QX, Indices{index::Q, index::i, index::a}, Qov, Indices{index::i, index::a, index::X}, ia_X);
    }
}