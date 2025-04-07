#ifndef _TBLIS_TEST_HPP_
#define _TBLIS_TEST_HPP_

#include <algorithm>
#include <limits>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <iomanip>
#include <map>
#include <typeinfo>
#include <cxxabi.h>
#include <chrono>
#include <list>

#include "tblis.h"
#include "util/random.hpp"
#include "util/tensor.hpp"
#include "util/macros.h"
#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/iostream.hpp"
#include "internal/3t/dense/mult.hpp"
#include "internal/3t/dpd/mult.hpp"

#include "external/catch/catch.hpp"

using namespace std;
using namespace stl_ext;
using namespace tblis;
using namespace tblis::internal;
using namespace tblis::detail;
using namespace tblis::slice;

#define INFO_OR_PRINT(...) INFO(__VA_ARGS__); //cout << __VA_ARGS__ << endl;

#define TENSOR_INFO(t) \
INFO_OR_PRINT("len_" #t "    = " << t.lengths()); \
INFO_OR_PRINT("stride_" #t " = " << t.strides()); \
INFO_OR_PRINT("idx_" #t "    = " << idx_##t);

#define DPD_TENSOR_INFO(t) \
INFO_OR_PRINT("irrep_" #t " = " << t.irrep()); \
INFO_OR_PRINT("len_" #t "   = \n" << t.lengths()); \
INFO_OR_PRINT("idx_" #t "   = " << idx_##t);

#define INDEXED_TENSOR_INFO(t) \
INFO_OR_PRINT("dense len_" #t "    = " << t.dense_lengths()); \
INFO_OR_PRINT("dense stride_" #t " = " << t.dense_strides()); \
INFO_OR_PRINT("idx len_" #t "      = " << t.indexed_lengths()); \
INFO_OR_PRINT("data_" #t "         = \n" << t.data()); \
INFO_OR_PRINT("indices_" #t "      = \n" << t.indices()); \
INFO_OR_PRINT("idx_" #t "          = " << substr(idx_##t,0,t.dense_dimension()) << \
                                   " " << substr(idx_##t,t.dense_dimension()));

#define INDEXED_DPD_TENSOR_INFO(t) \
INFO_OR_PRINT("irrep_" #t "       = " << t.irrep()); \
INFO_OR_PRINT("dense irrep_" #t " = " << t.dense_irrep()); \
INFO_OR_PRINT("dense len_" #t "   = \n" << t.dense_lengths()); \
INFO_OR_PRINT("idx irrep_" #t "   = " << t.indexed_irreps()); \
INFO_OR_PRINT("idx len_" #t "     = " << t.indexed_lengths()); \
INFO_OR_PRINT("nidx_" #t "        = " << t.num_indices()); \
INFO_OR_PRINT("data_" #t "        = \n" << t.data()); \
INFO_OR_PRINT("indices_" #t "     = \n" << t.indices()); \
INFO_OR_PRINT("idx_" #t "          = " << substr(idx_##t,0,t.dense_dimension()) << \
                                   " " << substr(idx_##t,t.dense_dimension()));

#define PRINT_TENSOR(t) \
cout << "\n" #t ":\n"; \
t.for_each_element( \
[](const typename decltype(t)::value_type & e, const index_vector& pos) \
{ \
    if (std::abs(e) > 1e-13) cout << pos << " " << e << endl; \
});

template <typename T>
auto data(const marray<T>& v) { return v.data(); }

template <typename T>
auto data(const dpd_marray<T>& v) { return v.data(); }

template <typename T>
auto data(const indexed_marray<T>& v) { return v.data(0); }

template <typename T>
auto data(const indexed_dpd_marray<T>& v) { return v.data(0); }

template <typename... Ts>
auto substr(const std::string& s, Ts&&... args)
{
    label_vector l;
    return s.substr(std::forward<Ts>(args)...);
}

template <size_t N>
auto substr(const MArray::short_vector<label_type,N>& s, int begin)
{
    MArray::short_vector<label_type,N> r;
    if (begin < s.size())
        r.insert(r.end(), s.begin()+begin, s.end());
    return r;
}

template <size_t N>
auto substr(const MArray::short_vector<label_type,N>& s, int begin, int len)
{
    MArray::short_vector<label_type,N> r;
    if (begin < s.size())
    {
        if (begin+len < s.size())
            r.insert(r.end(), s.begin()+begin, s.begin()+begin+len);
        else
            r.insert(r.end(), s.begin()+begin, s.end());
    }
    return r;
}

#define PRINT_DPD_TENSOR(t) \
cout << "\n" #t ":\n"; \
t.for_each_element( \
[&t](const typename decltype(t)::value_type & e, const irrep_vector& irreps, const index_vector& pos) \
{ \
    if (std::abs(e) > 1e-13) cout << irreps << " " << pos << " " << e << " " << (&e - data(t)) << endl; \
});

template <typename T>
void randomize_tensor(T& t)
{
    typedef typename T::value_type U;
    t.for_each_element([](U& e) { e = random_unit<U>(); });
}

template <typename T> const string& type_name();

template <typename... Types> struct types;

template <template <typename> class Body, typename... Types> struct templated_test_case_runner;

template <template <typename> class Body, typename... Types>
struct templated_test_case_runner<Body, types<Types...>>
{
    static void run()
    {
        templated_test_case_runner<Body, Types...>::run();
    }
};

template <template <typename> class Body, typename Type, typename... Types>
struct templated_test_case_runner<Body, Type, Types...>
{
    static void run()
    {
        {
            INFO_OR_PRINT("Template parameter: " << type_name<Type>());
            Body<Type>::run();
        }
        templated_test_case_runner<Body, Types...>::run();
    }
};

template <template <typename> class Body>
struct templated_test_case_runner<Body>
{
    static void run() {}
};

#define REPLICATED_TEST_CASE(name, ntrial) \
static void TBLIS_PASTE(__replicated_test_case_body_, name)(); \
TEST_CASE(#name) \
{ \
    for (int trial = 0;trial < ntrial;trial++) \
    { \
        INFO_OR_PRINT("Trial " << (trial+1) << " of " << ntrial); \
        TBLIS_PASTE(__replicated_test_case_body_, name)(); \
    } \
} \
static void TBLIS_PASTE(__replicated_test_case_body_, name)()

#define TEMPLATED_TEST_CASE(name, T, ...) \
template <typename T> struct TBLIS_PASTE(__templated_test_case_body_, name) \
{ \
    static void run(); \
}; \
TEST_CASE(#name) \
{ \
    templated_test_case_runner<TBLIS_PASTE(__templated_test_case_body_, name), __VA_ARGS__>::run(); \
} \
template <typename T> void TBLIS_PASTE(__templated_test_case_body_, name)<T>::run()

#define REPLICATED_TEMPLATED_TEST_CASE(name, ntrial, T, ...) \
template <typename T> static void TBLIS_PASTE(__replicated_templated_test_case_body_, name)(); \
TEMPLATED_TEST_CASE(name, T, __VA_ARGS__) \
{ \
    for (int trial = 0;trial < ntrial;trial++) \
    { \
        INFO_OR_PRINT("Trial " << (trial+1) << " of " << ntrial); \
        TBLIS_PASTE(__replicated_templated_test_case_body_, name)<T>(); \
    } \
} \
template <typename T> static void TBLIS_PASTE(__replicated_templated_test_case_body_, name)()

constexpr static int ulp_factor = 32;

extern stride_type N;
extern int R;
typedef types<float, double, scomplex, dcomplex> all_types;

enum index_type
{
    TYPE_A,
    TYPE_B,
    TYPE_C,
    TYPE_AB,
    TYPE_AC,
    TYPE_BC,
    TYPE_ABC
};

template <typename T>
len_vector group_size(const matrix<len_type>& len, const T& idx, const T& choose)
{
    unsigned nirrep = len.length(1);
    matrix<len_type> sublen({(unsigned)choose.size(), nirrep});

    for (unsigned i = 0;i < choose.size();i++)
    {
        for (unsigned j = 0;j < idx.size();j++)
        {
            if (choose[i] == idx[j])
            {
                sublen[i][0] = len[j][0];
            }
        }
    }

    len_vector size(nirrep);
    for (unsigned i = 0;i < nirrep;i++)
    {
        size[i] = dpd_marray<double>::size(i, sublen);
    }

    return size;
}

template <typename T>
double ceil2(T x)
{
    return nearbyint(pow(2.0, max(0.0, ceil(log2((double)std::abs(x))))));
}

template <typename T, typename U>
void check(const string& label, stride_type ia, stride_type ib, T error, U ulps)
{
    typedef decltype(std::abs(error)) V;
    auto epsilon = std::abs(max(numeric_limits<V>::min(),
       float(ceil2(ulp_factor*std::abs(ulps)))*numeric_limits<V>::epsilon()));

    INFO_OR_PRINT(label);
    INFO_OR_PRINT("Error = " << std::abs(error));
    INFO_OR_PRINT("Epsilon = " << epsilon);
    REQUIRE(std::abs(error) == Approx(0).epsilon(0).margin(epsilon));
    REQUIRE(ia == ib);
}

template <typename T, typename U>
void check(const string& label, T error, U ulps)
{
    check(label, 0, 0, error, ulps);
}

template <typename T, typename U, typename V>
void check(const string& label, stride_type ia, stride_type ib, T a, U b, V ulps)
{
    INFO_OR_PRINT("Values = " << a << ", " << b);
    check(label, ia, ib, a-b, ulps);
}

template <typename T, typename U, typename V>
void check(const string& label, T a, U b, V ulps)
{
    check(label, 0, 0, a, b, ulps);
}

template <typename T>
void gemm_ref(T alpha, matrix_view<const T> A,
                       matrix_view<const T> B,
              T  beta,       matrix_view<T> C);

template <typename T>
void gemm_ref(T alpha, matrix_view<const T> A,
                          row_view<const T> D,
                       matrix_view<const T> B,
              T  beta,       matrix_view<T> C);

/*
 * Creates a matrix whose total storage size is between N/4
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/16 and N/4. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void random_matrix(stride_type N, len_type m_min, len_type n_min, matrix<T>& t);

/*
 * Creates a matrix, whose total storage size is between N/4
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/16 and N/4. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void random_matrix(stride_type N, matrix<T>& t);

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, const vector<len_type>& len_min, len_vector& len);

matrix<len_type> random_indices(const len_vector& len, double sparsity);

template <typename T>
void random_tensor(stride_type N, unsigned d, const vector<len_type>& len_min, marray<T>& A);

template <typename T>
void random_tensor(stride_type N, unsigned d, unsigned nirrep, const vector<len_type>& len_min, dpd_marray<T>& A);

template <typename T>
void random_tensor(stride_type N, unsigned d, const vector<len_type>& len_min, indexed_marray<T>& A);

template <typename T>
void random_tensor(stride_type N, unsigned d, unsigned nirrep, const vector<len_type>& len_min, indexed_dpd_marray<T>& A);

template <typename T>
void random_tensor(stride_type N, unsigned d, const vector<len_type>& len_min, dpd_marray<T>& A);

template <typename T>
void random_tensor(stride_type N, unsigned d, const vector<len_type>& len_min, indexed_dpd_marray<T>& A);

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4 and N/2. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, len_vector& len);

template <typename T>
void random_tensor(stride_type N, unsigned d, T& A)
{
    random_tensor(N, d, vector<len_type>(d), A);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
void random_lengths(stride_type N, len_vector& len);

template <typename T>
void random_tensor(stride_type N, T& A)
{
    random_tensor(N, random_number(1,8), A);
}

void random_lengths(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    len_vector& len_A, label_vector& idx_A,
                    len_vector& len_B, label_vector& idx_B);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    marray<T>& A, label_vector& idx_A,
                    marray<T>& B, label_vector& idx_B);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    dpd_marray<T>& A, label_vector& idx_A,
                    dpd_marray<T>& B, label_vector& idx_B);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    indexed_marray<T>& A, label_vector& idx_A,
                    indexed_marray<T>& B, label_vector& idx_B);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    indexed_dpd_marray<T>& A, label_vector& idx_A,
                    indexed_dpd_marray<T>& B, label_vector& idx_B);

void random_lengths(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    len_vector& len_A, label_vector& idx_A,
                    len_vector& len_B, label_vector& idx_B,
                    len_vector& len_C, label_vector& idx_C);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    marray<T>& A, label_vector& idx_A,
                    marray<T>& B, label_vector& idx_B,
                    marray<T>& C, label_vector& idx_C);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    dpd_marray<T>& A, label_vector& idx_A,
                    dpd_marray<T>& B, label_vector& idx_B,
                    dpd_marray<T>& C, label_vector& idx_C);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    indexed_marray<T>& A, label_vector& idx_A,
                    indexed_marray<T>& B, label_vector& idx_B,
                    indexed_marray<T>& C, label_vector& idx_C);

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    indexed_dpd_marray<T>& A, label_vector& idx_A,
                    indexed_dpd_marray<T>& B, label_vector& idx_B,
                    indexed_dpd_marray<T>& C, label_vector& idx_C);

#endif
