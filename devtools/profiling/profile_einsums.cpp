//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities.hpp>

#include <vector>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace std;

void create_J(Tensor<double, 2> &J, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI) {
    LabeledSection0();
    einsum(0.0, Indices{index::mu, index::nu}, &J, 2.0, Indices{index::mu, index::nu, index::lambda, index::sigma}, TEI,
           Indices{index::lambda, index::sigma}, D);
}

void create_K(Tensor<double, 2> &K, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI) {
    LabeledSection0();
    einsum(0.0, Indices{index::mu, index::nu}, &K, -1.0, Indices{index::mu, index::lambda, index::nu, index::sigma}, TEI,
           Indices{index::lambda, index::sigma}, D);
}

void create_K_sorted(Tensor<double, 2> &K, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI, Tensor<double, 4> &sorted_TEI) {
    LabeledSection0();
    einsum(0.0, Indices{index::mu, index::nu}, &K, -1.0, Indices{index::mu, index::nu, index::lambda, index::sigma}, sorted_TEI,
           Indices{index::lambda, index::sigma}, D);
}

void create_G(Tensor<double, 2> &G, Tensor<double, 2> const &J, Tensor<double, 2> const &K) {
    LabeledSection0();
    G = J;
    G += K;
}

double mean(std::vector<double> const &values) {
    double out        = 0;
    size_t num_values = values.size();

    for (size_t i = 0; i < num_values; i++) {
        out += values[i];
    }

    return out / num_values;
}

double variance(std::vector<double> const &values, double mean) {
    double out        = 0;
    size_t num_values = values.size();

    for (size_t i = 0; i < num_values; i++) {
        out += (values[i] - mean) * (values[i] - mean);
    }

    return out / (num_values - 1);
}

double stdev(std::vector<double> const &values, double mean) {
    return sqrt(variance(values, mean));
}

void register_args() {
    auto &global_config = einsums::GlobalConfigMap::get_singleton();
    auto &global_ints   = global_config.get_int_map()->get_value();
    auto &global_bools  = global_config.get_bool_map()->get_value();

    static cl::Opt<int64_t> nStartingOrbitals{"start-norbitals",
                                              {'n'},
                                              "The starting number of orbitals for the calculation",
                                              cl::Default<int64_t>(20),
                                              cl::ValueName("n"),
                                              cl::Location(global_ints["n"])};
    static cl::Opt<int64_t> stepValue{"step",
                                      {'s'},
                                      "The step value for the range of orbitals.",
                                      cl::Default<int64_t>(10),
                                      cl::ValueName("s"),
                                      cl::Location(global_ints["s"])};
    static cl::Opt<int64_t> nEndingOrbitals{"end-norbitals",
                                            {'n'},
                                            "The ending number of orbitals for the calculation",
                                            cl::Default<int64_t>(-1),
                                            cl::ValueName("e"),
                                            cl::Location(global_ints["e"])};

    static cl::Opt<int64_t> nTrails{"ntrials",
                                    {'t'},
                                    "The number of trials for each step inthe calculation",
                                    cl::Default<int64_t>(20),
                                    cl::ValueName("t"),
                                    cl::Location(global_ints["t"])};

    static cl::Flag csv{"csv", {'c'}, "Print csv", cl::Location(global_bools["c"])};
}

int main(int argc, char **argv) {
#pragma omp parallel
    {
#pragma omp single
        {
            einsums::register_arguments(register_args);

            // Initialize einsums.
            einsums::initialize(argc, argv);

            int  start, end, step, trials;
            bool csv;

            {
                auto &global_config = einsums::GlobalConfigMap::get_singleton();

                start  = global_config.get_int("n");
                end    = global_config.get_int("e");
                step   = global_config.get_int("s");
                trials = global_config.get_int("t");
                csv    = global_config.get_bool("c");
            }

            if (end < start) {
                end = start;
            }

            for (int norbs = start; norbs <= end; norbs += step) {
                if (!csv) {
                    printf("Running %d trials with %d orbitals.\n", trials, norbs);
                }

                std::vector<double> times_J(trials), times_K(trials), times_G(trials), times_tot(trials), times_sort(trials);

                auto D   = create_random_tensor("D", norbs, norbs);
                auto TEI = create_random_tensor("TEI", norbs, norbs, norbs, norbs);

                Tensor<double, 2> J{"J", norbs, norbs}, K{"K", norbs, norbs}, G{"G", norbs, norbs};

                double J_mean;
                double K_mean;
                double G_mean;
                double tot_mean;

                // Calculate the times.
                {
                    LabeledSection("Unsorted");
                    for (int i = 0; i < trials; i++) {
                        clock_t const start = clock();

                        create_J(J, D, TEI);

                        clock_t const J_time = clock();

                        create_K(K, D, TEI);

                        clock_t const K_time = clock();

                        create_G(G, J, K);

                        clock_t const G_time = clock();

                        times_J[i]   = (J_time - start) / (double)CLOCKS_PER_SEC;
                        times_K[i]   = (K_time - J_time) / (double)CLOCKS_PER_SEC;
                        times_tot[i] = (G_time - start) / (double)CLOCKS_PER_SEC;
                        times_G[i]   = (G_time - K_time) / (double)CLOCKS_PER_SEC;
                    }

                    // Print the timing info.
                    J_mean   = mean(times_J);
                    K_mean   = mean(times_K);
                    G_mean   = mean(times_G);
                    tot_mean = mean(times_tot);

                    if (csv) {
                        printf("%d,%lf,%lf,", norbs, tot_mean, stdev(times_tot, tot_mean));
                    } else {
                        printf("einsums times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, stdev %lg s\ntotal: "
                               "%lg s, "
                               "stdev %lg s\n",
                               J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                               stdev(times_tot, tot_mean));
                    }
                }

                Tensor<double, 4> sorted_TEI{"sorted TEI", norbs, norbs, norbs, norbs};

                // Calculate the linear algebra times.
                {
                    LabeledSection("Sorted");
                    for (int i = 0; i < trials; i++) {
                        clock_t start = clock();

                        create_J(J, D, TEI);

                        clock_t J_time = clock();

                        tensor_algebra::permute(Indices{index::mu, index::nu, index::lambda, index::sigma}, &sorted_TEI,
                                                Indices{index::mu, index::lambda, index::nu, index::sigma}, TEI);

                        clock_t K_sort_time = clock();

                        create_K_sorted(K, D, TEI, sorted_TEI);

                        clock_t K_time = clock();

                        create_G(G, J, K);

                        clock_t G_time = clock();

                        times_J[i]    = (J_time - start) / (double)CLOCKS_PER_SEC;
                        times_K[i]    = (K_time - J_time) / (double)CLOCKS_PER_SEC;
                        times_tot[i]  = (G_time - start) / (double)CLOCKS_PER_SEC;
                        times_G[i]    = (G_time - K_time) / (double)CLOCKS_PER_SEC;
                        times_sort[i] = (K_sort_time - J_time) / (double)CLOCKS_PER_SEC;
                    }
                }

                // Print the timing info.
                J_mean           = mean(times_J);
                K_mean           = mean(times_K);
                G_mean           = mean(times_G);
                tot_mean         = mean(times_tot);
                double sort_mean = mean(times_sort);
                if (csv) {
                    printf("%lf,%lf\n", tot_mean, stdev(times_tot, tot_mean));
                } else {
                    printf("sorted times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, "
                           "stdev %lg "
                           "s\ntotal: %lg s, stdev %lg s\n",
                           J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                           stdev(times_tot, tot_mean));
                }
            }

            einsums::finalize();
        }
    }

    return 0;
}