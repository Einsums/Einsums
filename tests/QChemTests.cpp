#include <catch2/catch_all.hpp>
#include <cmath>
#include <initializer_list>

#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "einsums.hpp"

template <size_t Rank>
static void read_tensor(std::string fname, einsums::Tensor<double, Rank> *out) {
    std::FILE *input = std::fopen(fname.c_str(), "r");

    char buffer[1024] = {0};
    int  line_num     = 0;

    while (!std::feof(input)) {
        line_num++;
        std::memset(buffer, 0, 1024);
        std::fgets(buffer, 1023, input);
        std::array<int, Rank> indices;

        char *next = std::strtok(buffer, " \t");

        if (next == NULL) {
            continue;
        }

        indices[0] = std::atoi(next) - 1;

        for (int i = 1; i < Rank; i++) {
            next = std::strtok(NULL, " \t");

            if (next == NULL) {
                std::printf("Line %d in file ", line_num);
                println(fname);
                throw std::runtime_error("Line in file not formatted correctly!");
            }

            indices[i] = std::atoi(next) - 1;
        }

        next = std::strtok(NULL, " \t");

        if (next == NULL) {
            std::printf("Line %d in file ", line_num);
            println(fname);
            throw std::runtime_error("Line in file not formatted correctly!");
        }

        if constexpr (Rank == 2) {
            double val                = std::atof(next);
            std::apply(*out, indices) = val;
            std::swap(indices[0], indices[1]);
            std::apply(*out, indices) = val;
        } else if constexpr (Rank == 4) {
            double val = std::atof(next);

            std::array<int, 4> ind1{indices[0], indices[1], indices[2], indices[3]}, ind2{indices[1], indices[0], indices[2], indices[3]},
                ind3{indices[0], indices[1], indices[3], indices[2]}, ind4{indices[1], indices[0], indices[2], indices[3]},
                ind5{indices[2], indices[3], indices[0], indices[1]}, ind6{indices[3], indices[2], indices[0], indices[1]},
                ind7{indices[2], indices[3], indices[1], indices[0]}, ind8{indices[3], indices[2], indices[1], indices[0]};

            std::apply(*out, ind1) = val;
            std::apply(*out, ind2) = val;
            std::apply(*out, ind3) = val;
            std::apply(*out, ind4) = val;
            std::apply(*out, ind5) = val;
            std::apply(*out, ind6) = val;
            std::apply(*out, ind7) = val;
            std::apply(*out, ind8) = val;
        }
    }

    std::fclose(input);
}

static void update_Cocc(const einsums::Tensor<double, 1> &energies, einsums::BlockTensor<double, 2> *Cocc,
                        const einsums::BlockTensor<double, 2> &C, std::array<int, 4> &occ_per_irrep) {
    // Update occupation.

    std::array<int, 4> irrep_sizes{4, 0, 1, 2};

    for (int i = 0; i < 4; i++) {
        occ_per_irrep[i] = 0;
    }

    for (int i = 0; i < 5; i++) {
        double curr_min  = INFINITY;
        int    irrep_occ = -1;

        for (int j = 0; j < 4; j++) {
            if (occ_per_irrep[j] >= irrep_sizes[j]) {
                continue;
            }

            double energy = energies(C.block_range(j)[0] + occ_per_irrep[j]);

            if (energy < curr_min) {
                curr_min  = energy;
                irrep_occ = j;
            }
        }

        occ_per_irrep[irrep_occ]++;
    }

    (*Cocc).zero();
#pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        (*Cocc)[i](einsums::AllT{}, einsums::Range(0, occ_per_irrep[i])) = C[i](einsums::AllT{}, einsums::Range(0, occ_per_irrep[i]));
    }
}

static void compute_diis_coefs(const std::vector<einsums::BlockTensor<double, 2>> &errors, std::vector<double> *out) {
    einsums::Tensor<double, 2> *B_mat = new einsums::Tensor<double, 2>("DIIS error matrix", errors.size() + 1, errors.size() + 1);

    B_mat->zero();
    (*B_mat)(einsums::Range{errors.size(), errors.size() + 1}, einsums::Range{0, errors.size()}) = 1.0;
    (*B_mat)(einsums::Range{0, errors.size()}, einsums::Range{errors.size(), errors.size() + 1}) = 1.0;

#pragma omp parallel for
    for (int i = 0; i < errors.size(); i++) {
#pragma omp parallel for
        for (int j = 0; j <= i; j++) {
            (*B_mat)(i, j) = einsums::linear_algebra::dot(errors[i], errors[j]);
            (*B_mat)(j, i) = (*B_mat)(i, j);
        }
    }

    einsums::Tensor<double, 2> res_mat("DIIS result matrix", 1, errors.size() + 1);

    res_mat.zero();
    res_mat(0, errors.size()) = 1.0;

    einsums::linear_algebra::gesv(B_mat, &res_mat);

    out->resize(errors.size());

    for (int i = 0; i < errors.size(); i++) {
        out->at(i) = res_mat(0, i);
    }

    delete B_mat;
}

static void compute_diis_fock(const std::vector<double> &coefs, const std::vector<einsums::BlockTensor<double, 2>> &focks,
                              einsums::BlockTensor<double, 2> *out) {

    out->zero();

    for (int i = 0; i < coefs.size(); i++) {
        einsums::linear_algebra::axpy(coefs[i], focks[i], out);
    }
}

TEST_CASE("RHF") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::linear_algebra;

    Tensor<double, 2> S("Overlap", 7, 7), T("Kinetic Energy", 7, 7), V("Potential Energy", 7, 7), H("Core Hamiltonian", 7, 7),
        Symm("Symmetrizer", 7, 7), symm_temp1("Symmetrizing temp1", 7, 7), symm_temp2("Symmetrizing temp1", 7, 7);
    Tensor<double, 4> TEI("Two-electron integrals", 7, 7, 7, 7), TEI_temp1("Two-electron symmetrize temp1", 7, 7, 7, 7),
        TEI_temp2("Two-electron symmetrize temp2", 7, 7, 7, 7), TEI_temp3("Two-electron symmetrize temp3", 7, 7, 7, 7);

    BlockTensor<double, 2> S_sym("Overlap", 4, 0, 1, 2), H_sym("Hamiltonian", 4, 0, 1, 2), D("Density", 4, 0, 1, 2),
        D_prev("Previous density", 4, 0, 1, 2), F("Fock Matrix", 4, 0, 1, 2), Ft("Transformed Fock", 4, 0, 1, 2),
        X("Unitary Transform", 4, 0, 1, 2), C("MO coefs", 4, 0, 1, 2), Cocc("Occupied MO coefs", 4, 0, 1, 2),
        Ct("Transformed MO coefficients", 4, 0, 1, 2), temp1("Temp1", 4, 0, 1, 2), temp2("Temp2", 4, 0, 1, 2), FDS("FDS", 4, 0, 1, 2),
        SDF("SDF", 4, 0, 1, 2);
    std::vector<BlockTensor<double, 2>> DIIS_errors, DIIS_focks;
    std::vector<double>                 DIIS_coefs;

    TiledTensor<double, 4> TEI_sym("Two-electron integrals", {4, 0, 1, 2});

    Tensor<double, 1> Evals("Eigenvalues", 7);

    std::array<int, 4> occ_per_irrep;

    int cycles = 0;

    double e0 = -1, e1 = 0;
    double dRMS = 1;

    // Read in the values.
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/S.dat", &S));
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/T.dat", &T));
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/V.dat", &V));
    TEI.zero();
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/TEI.dat", &TEI));

    double enuc = 8.002367061810450;

    // Matrix for forming salcs.
    std::vector<double> symm_values{1, 0, 0, 0, 0, 0, 0,         0, 1, 0,         0, 0, 0, 0,         0, 0,
                                    0, 0, 0, 1, 0, 0, 0,         1, 0, 0,         0, 0, 0, 0,         0, 0,
                                    1, 0, 0, 0, 0, 0, M_SQRT1_2, 0, 0, -M_SQRT1_2, 0, 0, 0, M_SQRT1_2, 0, 0, M_SQRT1_2};

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            Symm(i, j) = symm_values[i * 7 + j];
        }
    }

    // Compute the Hamiltonian.
    H = T;
    H += V;

    // Symmetrize
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &symm_temp1, Indices{index::i, index::k}, S, Indices{index::k, index::j}, Symm));
    REQUIRE_NOTHROW(
        einsum(Indices{index::i, index::j}, &symm_temp2, Indices{index::k, index::i}, Symm, Indices{index::k, index::j}, symm_temp1));

    S_sym[0]       = symm_temp2(Range{0, 4}, Range{0, 4});
    S_sym[2](0, 0) = symm_temp2(4, 4);
    S_sym[3]       = symm_temp2(Range{5, 7}, Range{5, 7});

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &symm_temp1, Indices{index::i, index::k}, H, Indices{index::k, index::j}, Symm));
    REQUIRE_NOTHROW(
        einsum(Indices{index::i, index::j}, &symm_temp2, Indices{index::k, index::i}, Symm, Indices{index::k, index::j}, symm_temp1));

    H_sym[0]       = symm_temp2(Range{0, 4}, Range{0, 4});
    H_sym[2](0, 0) = symm_temp2(4, 4);
    H_sym[3]       = symm_temp2(Range{5, 7}, Range{5, 7});

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp1, Indices{index::i, index::m, index::k, index::l},
                           TEI, Indices{index::m, index::j}, Symm));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp2, Indices{index::m, index::j, index::k, index::l},
                           TEI_temp1, Indices{index::m, index::i}, Symm));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp1, Indices{index::i, index::j, index::m, index::l},
                           TEI_temp2, Indices{index::m, index::k}, Symm));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp2, Indices{index::i, index::j, index::k, index::m},
                           TEI_temp1, Indices{index::m, index::l}, Symm));

    TEI_sym.tile(0, 0, 0, 0) = TEI_temp2(Range{0, 4}, Range{0, 4}, Range{0, 4}, Range{0, 4});
    TEI_sym.tile(0, 0, 2, 2) = TEI_temp2(Range{0, 4}, Range{0, 4}, Range{4, 5}, Range{4, 5});
    TEI_sym.tile(0, 0, 3, 3) = TEI_temp2(Range{0, 4}, Range{0, 4}, Range{5, 7}, Range{5, 7});
    TEI_sym.tile(2, 2, 0, 0) = TEI_temp2(Range{4, 5}, Range{4, 5}, Range{0, 4}, Range{0, 4});
    TEI_sym.tile(2, 2, 2, 2) = TEI_temp2(4, 4, 4, 4);
    TEI_sym.tile(2, 2, 3, 3) = TEI_temp2(Range{4, 5}, Range{4, 5}, Range{5, 7}, Range{5, 7});
    TEI_sym.tile(3, 3, 0, 0) = TEI_temp2(Range{5, 7}, Range{5, 7}, Range{0, 4}, Range{0, 4});
    TEI_sym.tile(3, 3, 2, 2) = TEI_temp2(Range{5, 7}, Range{5, 7}, Range{4, 5}, Range{4, 5});
    TEI_sym.tile(3, 3, 3, 3) = TEI_temp2(Range{5, 7}, Range{5, 7}, Range{5, 7}, Range{5, 7});

    // Compute the unitary transform.
    X = einsums::linear_algebra::pow(S_sym, -0.5);

    // Compute the guess Fock matrix.
    F = H_sym;

    // Transform.
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, X));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &Ft, Indices{index::k, index::j}, temp1, Indices{index::k, index::i}, X));

    // Compute the coefficients.
    Ct = Ft;
    syev(&Ct, &Evals);

    // Transform back.
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::k, index::j}, Ct, Indices{index::i, index::k}, X));

    // Compute the occupied orbitals.
    update_Cocc(Evals, &Cocc, C, occ_per_irrep);

    // Compute the density matrix.
    einsum(Indices{index::i, index::j}, &D, Indices{index::i, index::m}, Cocc, Indices{index::j, index::m}, Cocc);

    e0 = enuc;

    Tensor<double, 0> Elec;

    REQUIRE_NOTHROW(einsum(Indices{}, &Elec, Indices{index::i, index::j}, F, Indices{index::i, index::j}, D));
    REQUIRE_NOTHROW(einsum(1.0, Indices{}, &Elec, 1.0, Indices{index::i, index::j}, H_sym, Indices{index::i, index::j}, D));

    e0 += (double)Elec;

    println(H_sym);
    println(S_sym);
    println(X);
    println(C);
    println(D);

    printf("Initial occupation: ");
    for(auto val : occ_per_irrep) {
        printf("%d ", val);
    }

    printf("\n%d\t%lf\n", 0, e0);

    while (std::fabs(e0 - e1) > 1e-10 && dRMS > 1e-10 && cycles < 50) {
        D_prev = D;
        e1     = e0;

        // Compute the new Fock matrix.
        F = H_sym;

        REQUIRE_NOTHROW(einsum(1.0, Indices{index::i, index::j}, &F, 2.0, Indices{index::i, index::j, index::k, index::l}, TEI_sym,
                               Indices{index::k, index::l}, D));
        REQUIRE_NOTHROW(einsum(1.0, Indices{index::i, index::j}, &F, -1.0, Indices{index::i, index::k, index::j, index::l}, TEI_sym,
                               Indices{index::k, index::l}, D));

        // Transform.
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, X));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &Ft, Indices{index::k, index::j}, temp1, Indices{index::k, index::i}, X));

        // Compute the coefficients.
        Ct = Ft;
        syev(&Ct, &Evals);

        // Transform back.
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::k, index::j}, Ct, Indices{index::i, index::k}, X));

        // Compute the occupied orbitals.
        update_Cocc(Evals, &Cocc, C, occ_per_irrep);

        // Compute the density matrix.
        einsum(Indices{index::i, index::j}, &D, Indices{index::i, index::m}, Cocc, Indices{index::j, index::m}, Cocc);

        // Compute the new energy.
        e0 = enuc;

        REQUIRE_NOTHROW(einsum(Indices{}, &Elec, Indices{index::i, index::j}, F, Indices{index::i, index::j}, D));
        REQUIRE_NOTHROW(einsum(1.0, Indices{}, &Elec, 1.0, Indices{index::i, index::j}, H_sym, Indices{index::i, index::j}, D));

        e0 += (double)Elec;

        if (cycles > 0) {
            Tensor<double, 0> rms;

            temp1 = D;
            temp1 -= D_prev;

            REQUIRE_NOTHROW(einsum(Indices{}, &rms, Indices{index::i, index::j}, temp1, Indices{index::i, index::j}, temp2));

            dRMS = std::sqrt((double)rms);
        }

        printf("%d\t%lf\t%lf\t%lf\n", cycles, e0, e0 - e1, dRMS);

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, S_sym, Indices{index::k, index::j}, D));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &SDF, Indices{index::i, index::k}, temp1, Indices{index::k, index::j}, F));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, D));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &FDS, Indices{index::i, index::k}, temp1, Indices{index::k, index::j}, S_sym));

        temp1 = FDS;
        temp1 -= SDF;

        // if (DIIS_errors.size() == 6) {
        //     double max_error = -INFINITY;
        //     int    max_index = -1;

        //     Tensor<double, 0> dot_temp;

        //     for (int i = 0; i < DIIS_errors.size(); i++) {
        //         einsum(Indices{}, &dot_temp, Indices{index::i, index::j}, temp1, Indices{index::i, index::j}, DIIS_errors[i]);

        //         if ((double)dot_temp < max_error) {
        //             max_error = (double)dot_temp;
        //             max_index = i;
        //         }
        //     }

        //     DIIS_focks[max_index]  = F;
        //     DIIS_errors[max_index] = temp1;
        // } else {
        //     DIIS_errors.push_back(temp1);
        //     DIIS_focks.push_back(F);
        //     DIIS_coefs.push_back(0.0);
        // }

        // compute_diis_coefs(DIIS_errors, &DIIS_coefs);
        // compute_diis_fock(DIIS_coefs, DIIS_focks, &F);

        cycles++;
    }

    REQUIRE(cycles < 50);
    REQUIRE_THAT(e0, Catch::Matchers::WithinAbs(-74.942079928192, 1e-6));
}

TEST_CASE("RHF-MP2") {
}