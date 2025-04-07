#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <iostream>
#include <random>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <type_traits>
#include <iomanip>
#include <functional>
#include <set>
#include <map>
#include <atomic>

#include "tblis.h"
#include "util/time.hpp"
#include "util/tensor.hpp"
#include "util/random.hpp"
#include "internal/3t/dense/mult.hpp"

int check = 0;

using namespace std;
using namespace tblis;
using namespace stl_ext;

len_type v = 40;
len_type o = 20;
len_type g = 8;

namespace tblis
{

std::atomic<long> flops;
len_type inout_ratio = 200000;
int outer_threading = 1;

}

template <typename Kernel, typename ...Args>
double run_kernel(len_type R, Kernel&& kernel, Args&&...args)
{
    double bias = numeric_limits<double>::max();
    for (len_type r = 0;r < R;r++)
    {
        double t0 = tic();
        double t1 = tic();
        bias = min(bias, t1-t0);
    }

    double dt = numeric_limits<double>::max();
    for (len_type r = 0;r < R;r++)
    {
        double t0 = tic();
        kernel(args...);
        double t1 = tic();
        dt = min(dt, t1-t0);
    }

    return dt - bias;
}

template <typename T>
double diff(const dpd_marray_view<T>& A, const dpd_marray_view<T>& B)
{
    const T* a = A.data();
    const T* b = B.data();
    stride_type size = dpd_marray<T>::size(A.irrep(), A.lengths());

    double d = 0;

    for (stride_type i = 0;i < size;i++)
    {
        d += norm2(a[i]-b[i]);
    }

    return sqrt(d);
}

template <typename T>
void randomize(dpd_marray<T>& A)
{
    T* a = A.data();
    stride_type size = dpd_marray<T>::size(A.irrep(), A.lengths());

    for (stride_type i = 0;i < size;i++)
    {
        a[i] = random_number<double>();
    }
}

template <typename T>
void bench(int R,
           T alpha, const dpd_marray<T>& A, const std::string& typea,
                    const dpd_marray<T>& B, const std::string& typeb,
           T  beta,       dpd_marray<T>& C, const std::string& typec)
{
    dpd_marray<T> tmp1_, tmp2_;
    dpd_marray_view<T> tmp1, tmp2;

    if (check)
    {
        tmp1_.reset(C);
        tmp2_.reset(C);
        tmp1.reset(tmp1_);
        tmp2.reset(tmp2_);
    }
    else
    {
        tmp1.reset(C);
        tmp2.reset(C);
    }

    flops = 0;

    double t1 = run_kernel(R,
    [&]
    {
        internal::impl = internal::REFERENCE;
        mult<double>(alpha,    A, typea.data(),
                               B, typeb.data(),
                      beta, tmp1, typec.data());
    });

    auto flops1 = flops.load();
    printf("%ld\n", flops1);
    flops = 0;

    double t2 = run_kernel(R,
    [&]
    {
        internal::impl = internal::BLIS_BASED;
        mult<double>(alpha,    A, typea.data(),
                               B, typeb.data(),
                      beta, tmp2, typec.data());
    });

    auto flops2 = flops.load();
    printf("%ld\n", flops2);

    if (check)
    {
        double d = diff(tmp1, tmp2);
        printf("%g\n", d);
    }

    printf("%g %g\n", t1, t2);
    printf("%g %g\n", flops1/t1/1e9/R, flops2/t2/1e9/R);
}

int main(int argc, char** argv)
{
    int R = 5;
    time_t seed = time(nullptr);

    struct option opts[] = {{"rep", required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {"check", no_argument, &check, 1},
                            {"no-check", no_argument, &check, 0},
                            {"outer-threading", no_argument, &outer_threading, 1},
                            {"no-outer-threading", no_argument, &outer_threading, 0},
                            {"inner-threading", no_argument, &outer_threading, 0},
                            {"no-inner-threading", no_argument, &outer_threading, 1},
                            {"inout-ratio", required_argument, NULL, 'i'},
                            {"occ", required_argument, NULL, 'o'},
                            {"vrt", required_argument, NULL, 'v'},
                            {"nirrep", required_argument, NULL, 'g'},
                            {0, 0, 0, 0}};

    while (true)
    {
        istringstream iss;
        int arg = getopt_long(argc, argv, "r:s:v:o:i:g:", opts, NULL);

        if (arg == -1) break;

        switch (arg)
        {
            case 'i':
                iss.str(optarg);
                iss >> inout_ratio;
                break;
            case 'v':
                iss.str(optarg);
                iss >> v;
                break;
            case 'g':
                iss.str(optarg);
                iss >> g;
                break;
            case 'o':
                iss.str(optarg);
                iss >> o;
                break;
            case 'r':
                iss.str(optarg);
                iss >> R;
                break;
            case 's':
                iss.str(optarg);
                iss >> seed;
                break;
            case '?':
                abort();
                break;
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    rand_engine.seed(seed);

    len_vector vs(g,v);
    len_vector os(g,o);

    if (g > 1)
    {
        len_vector splits(g-1);
        for (len_type& split : splits) split = random_number<len_type>(v);
        sort(splits);

        vs[0] = splits[0];
        vs[g-1] = v-splits[g-2];
        for (unsigned i = 1;i < g-1;i++) vs[i] = splits[i]-splits[i-1];

        for (len_type& split : splits) split = random_number<len_type>(o);
        sort(splits);

        os[0] = splits[0];
        os[g-1] = o-splits[g-2];
        for (unsigned i = 1;i < g-1;i++) os[i] = splits[i]-splits[i-1];
    }

    for (unsigned i = 0;i < g;i++)
    {
        vs[i] = (v+g-1-i)/g;
        os[i] = (o+g-1-i)/g;
    }

    cout << "v: " << v << " -> " << vs[0];
    for (unsigned i = 1;i < g;i++) cout << ", " << vs[i];
    cout << endl;

    cout << "o: " << o << " -> " << os[0];
    for (unsigned i = 1;i < g;i++) cout << ", " << os[i];
    cout << endl;

    constexpr bool test0 = false;
    constexpr bool test1 = true;

    if (test0)
    {
        dpd_marray<double> A(0, g, {vs,vs});
        dpd_marray<double> B(0, g, {vs,vs});
        dpd_marray<double> C(0, g, {vs,vs});

        randomize(A);
        randomize(B);
        randomize(C);

        bench(R, 1.0, A, "AE",
                      B, "EB",
                 1.0, C, "AB");
    }

    if (test1)
    {
        dpd_marray<double> A(0, g, {vs,vs,os,os});
        dpd_marray<double> B(0, g, {vs,vs,os,os});
        dpd_marray<double> C(0, g, {vs,vs,os,os});

        randomize(A);
        randomize(B);
        randomize(C);

        bench(R, 1.0, A, "AEIM",
                      B, "EBMJ",
                 1.0, C, "ABIJ");
    }

    return 0;
}
