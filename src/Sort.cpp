#include "einsums/Sort.hpp"

#if defined(EINSUMS_USE_HPTT)
#    include "hptt/hptt.h"

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#    if defined(I)
#        undef I
#    endif
#endif

namespace einsums::tensor_algebra::detail {

void sort(const int *perm, const int dim, const float alpha, const float *A, const int *sizeA, const float beta, float *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void sort(const int *perm, const int dim, const double alpha, const double *A, const int *sizeA, const double beta, double *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void sort(const int *perm, const int dim, const std::complex<float> alpha, const std::complex<float> *A, const int *sizeA,
          const std::complex<float> beta, std::complex<float> *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

void sort(const int *perm, const int dim, const std::complex<double> alpha, const std::complex<double> *A, const int *sizeA,
          const std::complex<double> beta, std::complex<double> *B) {
    auto plan =
        hptt::create_plan(perm, dim, alpha, A, sizeA, nullptr, beta, B, nullptr, hptt::ESTIMATE, omp_get_max_threads(), nullptr, true);
    plan->execute();
}

} // namespace einsums::tensor_algebra::detail