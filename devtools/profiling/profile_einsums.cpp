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
    tensor_algebra::einsum(0.0, Indices{index::mu, index::nu}, &J, 2.0, Indices{index::mu, index::nu, index::lambda, index::sigma}, TEI,
                           Indices{index::lambda, index::sigma}, D);
}

void create_K(Tensor<double, 2> &K, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI) {
    tensor_algebra::einsum(0.0, Indices{index::mu, index::nu}, &K, -1.0, Indices{index::mu, index::lambda, index::nu, index::sigma}, TEI,
                           Indices{index::lambda, index::sigma}, D);
}

void create_K_sorted(Tensor<double, 2> &K, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI, Tensor<double, 4> &sorted_TEI) {
    tensor_algebra::einsum(0.0, Indices{index::mu, index::nu}, &K, -1.0, Indices{index::mu, index::nu, index::lambda, index::sigma},
                           sorted_TEI, Indices{index::lambda, index::sigma}, D);
}

void create_G(Tensor<double, 2> &G, Tensor<double, 2> const &J, Tensor<double, 2> const &K) {
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

static char const help[] = "Arguments:\n\n"
                           "-n NUMBER\t\tThe number of orbitals. Defaults to 20.\n\n"
                           "-t NUMBER\t\tThe number of trials. Defaults to 20.\n\n"
                           "-h, --help\t\tPrint the help message.";

void parse_args(int argc, char **argv, int *norbs, int *trials) {
    *norbs  = 20;
    *trials = 20;

    int state = 0; // 0 for no argument, 1 for orbs, 2 for trials.
    int i     = 1;

    while (i < argc) {
        if (state == 0) {
            if (strncmp(argv[i], "-n", 3) == 0) {
                state = 1;
            } else if (strncmp(argv[i], "-t", 3) == 0) {
                state = 2;
            } else if (strncmp(argv[i], "-h", 3) == 0 || strncmp(argv[i], "--help", 7) == 0) {
                puts(help);
                exit(0);
            } else {
                perror("Could not understand arguments! Please see the help message by using -h or --help. Please also make sure that you "
                       "put a space between a flag and its value.");
                exit(1);
            }
        } else if (state == 1) {
            int retval;

            retval = sscanf(argv[i], "%d", norbs);

            if (retval != 1) {
                perror("Could not understand input! Please put an argument after -n.");
                exit(1);
            }

            if (*norbs < 1) {
                perror("Cannot handle negative or zero numbers of orbitals!");
                exit(1);
            }
            state = 0;
        } else if (state == 2) {
            int retval;
            retval = sscanf(argv[i], "%d", trials);

            if (retval != 1) {
                perror("Could not understand input! Please put an argument after -t.");
                exit(1);
            }

            if (*trials < 1) {
                perror("Cannot handle negative or zero numbers of trials!");
                exit(1);
            }
            state = 0;
        } else {
            perror("Something went horribly wrong. Try again.");
            exit(-1);
        }
        i++;
    }
}

int main(int argc, char **argv) {
#pragma omp parallel
    {
#pragma omp single
        {
            // Initialize einsums.
            einsums::initialize(argc, argv);

            int norbs, trials;

            parse_args(argc, argv, &norbs, &trials);

            printf("Running %d trials with %d orbitals.\n", trials, norbs);

            std::vector<double> times_J(trials), times_K(trials), times_G(trials), times_tot(trials), times_sort(trials);

            auto D   = create_random_tensor("D", norbs, norbs);
            auto TEI = create_random_tensor("TEI", norbs, norbs, norbs, norbs);

            Tensor<double, 2> J{"J", norbs, norbs}, K{"K", norbs, norbs}, G{"G", norbs, norbs};

            // Calculate the times.
            for (int i = 0; i < trials; i++) {
                clock_t start = clock();

                create_J(J, D, TEI);

                clock_t J_time = clock();

                create_K(K, D, TEI);

                clock_t K_time = clock();

                create_G(G, J, K);

                clock_t G_time = clock();

                times_J[i]   = (J_time - start) / (double)CLOCKS_PER_SEC;
                times_K[i]   = (K_time - J_time) / (double)CLOCKS_PER_SEC;
                times_tot[i] = (G_time - start) / (double)CLOCKS_PER_SEC;
                times_G[i]   = (G_time - K_time) / (double)CLOCKS_PER_SEC;
            }

            // Print the timing info.
            double J_mean   = mean(times_J);
            double K_mean   = mean(times_K);
            double G_mean   = mean(times_G);
            double tot_mean = mean(times_tot);
            printf("einsums times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, stdev %lg s\ntotal: %lg s, "
                   "stdev %lg s\n",
                   J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                   stdev(times_tot, tot_mean));

            Tensor<double, 4> sorted_TEI{"sorted TEI", norbs, norbs, norbs, norbs};

            // Calculate the linear algebra times.
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

            // Print the timing info.
            J_mean           = mean(times_J);
            K_mean           = mean(times_K);
            G_mean           = mean(times_G);
            tot_mean         = mean(times_tot);
            double sort_mean = mean(times_sort);
            printf("sorted times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, "
                   "stdev %lg "
                   "s\ntotal: %lg s, stdev %lg s\n",
                   J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                   stdev(times_tot, tot_mean));

            einsums::finalize();
        }
    }

    return 0;
}