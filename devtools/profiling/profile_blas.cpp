#include <Einsums/BLAS.hpp>
#include <Einsums/Runtime.hpp>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace std;
using namespace einsums::blas;

void create_J(double *J, double const *D, double const *TEI, size_t norbs) {
    gemv('N', norbs * norbs, norbs * norbs, 2.0, TEI, norbs * norbs, D, 1, 0.0, J, 1);
}

void create_K(double *K, double const *D, double const *TEI, size_t norbs) {
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            K[i * norbs + j] = 0.0;
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    K[i * norbs + j] -= D[k * norbs + l] * TEI[norbs * (norbs * (norbs * i + k) + j) + l];
                }
            }
        }
    }
}

void create_K_sorted(double *K, double const *D, double const *TEI, double *sorted_TEI, size_t norbs) {
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    sorted_TEI[norbs * (norbs * (norbs * i + j) + k) + l] = TEI[norbs * (norbs * (norbs * i + k) + j) + l];
                }
            }
        }
    }

    gemv('N', norbs * norbs, norbs * norbs, -1.0, sorted_TEI, norbs * norbs, D, 1, 0.0, K, 1);
}

void create_G(double *G, double const *J, double const *K, size_t norbs) {
    memcpy((void *)G, (void const *)J, norbs * norbs * sizeof(double));
    axpy(norbs * norbs, 1.0, K, 1, G, 1);
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

template <class Generator>
void fill_random(std::vector<double> &buffer, Generator &generator) {
    std::uniform_real_distribution random_gen(-1.0, 1.0);
#pragma omp parallel for
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = random_gen(generator);
    }
}

int main(int argc, char **argv) {
#pragma omp parallel
    {
#pragma omp single
        {
            einsums::initialize(argc, argv);
            // Initialize random number generator.
            std::default_random_engine engine(clock());

            int norbs, trials;

            parse_args(argc, argv, &norbs, &trials);

            printf("Running %d trials with %d orbitals.\n", trials, norbs);

            std::vector<double> times_J(trials), times_K(trials), times_G(trials), times_tot(trials);

            std::vector<double> J(norbs * norbs), K(norbs * norbs), G(norbs * norbs), D(norbs * norbs), TEI(norbs * norbs * norbs * norbs);

            fill_random(D, engine);
            fill_random(TEI, engine);

            // Calculate the times.
            for (int i = 0; i < trials; i++) {
                clock_t start = clock();

                create_J(J.data(), D.data(), TEI.data(), norbs);

                clock_t J_time = clock();

                create_K(K.data(), D.data(), TEI.data(), norbs);

                clock_t K_time = clock();

                create_G(G.data(), J.data(), K.data(), norbs);

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

            std::vector<double> TEI_sorted(norbs * norbs * norbs * norbs);

            // Calculate the linear algebra times.
            for (int i = 0; i < trials; i++) {
                clock_t start = clock();

                create_J(J.data(), D.data(), TEI.data(), norbs);

                clock_t J_time = clock();

                create_K_sorted(K.data(), D.data(), TEI.data(), TEI_sorted.data(), norbs);

                clock_t K_time = clock();

                create_G(G.data(), J.data(), K.data(), norbs);

                clock_t G_time = clock();

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
            printf("sorted times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, stdev %lg s\ntotal: %lg s, stdev "
                   "%lg s\n",
                   J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                   stdev(times_tot, tot_mean));

            einsums::finalize();
        }
    }
    return 0;
}