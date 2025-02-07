#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

void create_J(double *__restrict__ J, double const *__restrict__ D, double const *__restrict__ TEI, size_t norbs) {
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            J[i * norbs + j] = 0.0;
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    J[i * norbs + j] += 2 * D[k * norbs + l] * TEI[norbs * (norbs * (norbs * i + j) + k) + l];
                }
            }
        }
    }
}

void create_K(double *__restrict__ K, double const *__restrict__ D, double const *__restrict__ TEI, size_t norbs) {
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

void create_G(double *__restrict__ G, double const *__restrict__ J, double const *__restrict__ K, size_t norbs) {
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            G[i * norbs + j] = J[i * norbs + j] + K[i * norbs + j];
        }
    }
}

void create_J_omp(double *__restrict__ J, double const *__restrict__ D, double const *__restrict__ TEI, size_t norbs) {
    memset(J, 0, norbs * sizeof(double));

#pragma omp parallel for simd collapse(4)
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    J[i * norbs + j] += 2 * D[k * norbs + l] * TEI[norbs * (norbs * (norbs * i + j) + k) + l];
                }
            }
        }
    }
}

void create_K_omp(double *__restrict__ K, double const *__restrict__ D, double const *__restrict__ TEI, size_t norbs) {
    memset(K, 0, norbs * sizeof(double));

#pragma omp parallel for simd collapse(4)
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    K[i * norbs + j] += 2 * D[k * norbs + l] * TEI[norbs * (norbs * (norbs * i + k) + j) + l];
                }
            }
        }
    }
}

void create_G_omp(double *__restrict__ G, double const *__restrict__ J, double const *__restrict__ K, size_t norbs) {
#pragma omp parallel for simd collapse(2)
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            G[i * norbs + j] = J[i * norbs + j] + K[i * norbs + j];
        }
    }
}

double mean(double const *__restrict__ values, size_t num_values) {
    double out = 0;

    for (size_t i = 0; i < num_values; i++) {
        out += values[i];
    }

    return out / num_values;
}

double variance(double const *__restrict__ values, size_t num_values, double mean) {
    double out = 0;

    for (size_t i = 0; i < num_values; i++) {
        out += (values[i] - mean) * (values[i] - mean);
    }

    return out / (num_values - 1);
}

double stdev(double const *__restrict__ values, size_t num_values, double mean) {
    return sqrt(variance(values, num_values, mean));
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

// Calculate a random number between -1 and 1.
double random_double() {
    union {
        uint64_t l;
        double   d;
    } value;

    uint64_t low  = rand();
    uint64_t high = rand();
    high <<= 31;

    value.l = ((low + high) & 0x000fffffffffffffUL) + 0x3ff0000000000000UL;
    if (high & 0x0010000000000000UL) {
        value.l |= 0x1000000000000000UL;
    }

    return value.d;
}

int main(int argc, char **argv) {
    // Seed the random number generator.
    struct timeval seed;
    gettimeofday(&seed, NULL);
    srand(seed.tv_sec ^ seed.tv_usec);

    int norbs, trials;

    double *times_J, *times_K, *times_G, *times_tot, *J, *K, *G, *TEI, *D;

    parse_args(argc, argv, &norbs, &trials);

    printf("Running %d trials with %d orbitals.\n", trials, norbs);

    times_J   = calloc(trials, sizeof(double));
    times_K   = calloc(trials, sizeof(double));
    times_G   = calloc(trials, sizeof(double));
    times_tot = calloc(trials, sizeof(double));
    J         = calloc(norbs * norbs, sizeof(double));
    K         = calloc(norbs * norbs, sizeof(double));
    G         = calloc(norbs * norbs, sizeof(double));
    D         = calloc(norbs * norbs, sizeof(double));
    TEI       = calloc(norbs * norbs * norbs * norbs, sizeof(double));

// Set up the density matrix.
    for (size_t i = 0; i < norbs * norbs; i++) {
        D[i] = random_double();
    }

// Set up the TEI matrix.
    for (size_t i = 0; i < norbs * norbs * norbs * norbs; i++) {
        TEI[i] = random_double();
    }

    // Calculate the times.
    for (int i = 0; i < trials; i++) {
        clock_t start = clock();

        create_J(J, D, TEI, norbs);

        clock_t J_time = clock();

        create_K(K, D, TEI, norbs);

        clock_t K_time = clock();

        create_G(G, J, K, norbs);

        clock_t G_time = clock();

        times_J[i]   = (J_time - start) / (double)CLOCKS_PER_SEC;
        times_K[i]   = (K_time - J_time) / (double)CLOCKS_PER_SEC;
        times_tot[i] = (G_time - start) / (double)CLOCKS_PER_SEC;
        times_G[i]   = (G_time - K_time) / (double)CLOCKS_PER_SEC;
    }

    // Print the timing info.
    double J_mean   = mean(times_J, trials);
    double K_mean   = mean(times_K, trials);
    double G_mean   = mean(times_G, trials);
    double tot_mean = mean(times_tot, trials);
    printf(
        "Non-omp times:\nform J: %lf s, stdev %lf s\nform K: %lf s, stdev %lf s\nform G: %lf s, stdev %lf s\ntotal: %lf s, stdev %lf s\n",
        J_mean, stdev(times_J, trials, J_mean), K_mean, stdev(times_K, trials, K_mean), G_mean, stdev(times_G, trials, G_mean), tot_mean,
        stdev(times_tot, trials, tot_mean));

    // Calculate the OMP times.
    for (int i = 0; i < trials; i++) {
        clock_t start = clock();

        create_J_omp(J, D, TEI, norbs);

        clock_t J_time = clock();

        create_K_omp(K, D, TEI, norbs);

        clock_t K_time = clock();

        create_G_omp(G, J, K, norbs);

        clock_t G_time = clock();

        times_J[i]   = (J_time - start) / (double)CLOCKS_PER_SEC;
        times_K[i]   = (K_time - J_time) / (double)CLOCKS_PER_SEC;
        times_tot[i] = (G_time - start) / (double)CLOCKS_PER_SEC;
        times_G[i]   = (G_time - K_time) / (double)CLOCKS_PER_SEC;
    }

    // Print the timing info.
    J_mean   = mean(times_J, trials);
    K_mean   = mean(times_K, trials);
    G_mean   = mean(times_G, trials);
    tot_mean = mean(times_tot, trials);
    printf("omp times:\nform J: %lf s, stdev %lf s\nform K: %lf s, stdev %lf s\nform G: %lf s, stdev %lf s\ntotal: %lf s, stdev %lf s\n",
           J_mean, stdev(times_J, trials, J_mean), K_mean, stdev(times_K, trials, K_mean), G_mean, stdev(times_G, trials, G_mean), tot_mean,
           stdev(times_tot, trials, tot_mean));

    free(J);
    free(K);
    free(G);
    free(D);
    free(TEI);
    free(times_J);
    free(times_K);
    free(times_G);
    free(times_tot);

    return 0;
}