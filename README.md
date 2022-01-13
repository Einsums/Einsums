# Einsums in C++

Provides compile-time contraction pattern analysis to determine optimal operation to perform.

## Examples
This will optimize at compile-time to a BLAS dgemm call.
```C++
#include "EinsumsInCpp/TensorAlgebra.hpp"

using EinsumsInCpp;  // Provides Tensor and create_random_tensor
using EinsumsInCpp::TensorAlgebra;  // Provides einsum and Indices
using EinsumsInCpp::TensorAlgrebra::Index;  // Provides i, j, k

Tensor<2> A = create_random_tensor("A", 7, 7);
Tensor<2> B = create_random_tensor("B", 7, 7);
Tensor<2> C{"C", 7, 7};

einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
```

Two-Electron Contribution to the Fock Matrix
```C++
#include "EinsumsInCpp/TensorAlgebra.hpp"

using namespace EinsumsInCpp;

void build_Fock_2e_einsum(Tensor<2> *F, 
                          const Tensor<4> &g,
                          const Tensor<2> &D) {
    using namespace EinsumsInCpp::TensorAlgebra;
    using namespace EinsumsInCpp::TensorAlgebra::Index;

    // Will compile-time optimize to BLAS gemv
    einsum(1.0, Indices{p, q}, F,
           2.0, Indices{p, q, r, s}, g, Indices{r, s}, D);

    // As written cannot be optimized.
    // A generic arbitrary contraction function will be used.
    einsum(1.0, Indices{p, q}, F, 
          -1.0, Indices{p, r, q, s}, g, Indices{r, s}, D);
}
```

![einsum Performance](/images/Performance.png)

W Intermediates in CCD
```C++
Wmnij = g_oooo;
// Compile-time optimizes to gemm
einsum(1.0,  Indices{m, n, i, j}, &Wmnij, 
       0.25, Indices{i, j, e, f}, t_oovv, 
             Indices{m, n, e, f}, g_oovv);

Wabef = g_vvvv;
// Compile-time optimizes to gemm
einsum(1.0,  Indices{a, b, e, f}, &Wabef, 
       0.25, Indices{m, n, e, f}, g_oovv,
             Indices{m, n, a, b}, t_oovv);

Wmbej = g_ovvo;
// As written uses generic arbitrary contraction function
einsum(1.0, Indices{m, b, e, j}, &Wmbej, 
      -0.5, Indices{j, n, f, b}, t_oovv, 
            Indices{m, n, e, f}, g_oovv);
```

CCD Energy
```C++
/// Compile-time optimizes to a dot product
einsum(0.0,  Indices{}, &e_ccd, 
       0.25, Indices{i, j, a, b}, new_t_oovv, 
             Indices{i, j, a, b}, g_oovv);
```
