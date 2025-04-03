# Einsums in C++

|   |   |
|---|---|
| **Status** | [![codecov](https://codecov.io/github/Einsums/Einsums/graph/badge.svg?token=Z8WA6CEGQA)](https://codecov.io/github/Einsums/Einsums) ![GitHub branch check runs](https://img.shields.io/github/check-runs/Einsums/Einsums/main) |
| **Release** | ![GitHub Release](https://img.shields.io/github/v/release/Einsums/Einsums) ![GitHub commits since latest release](https://img.shields.io/github/commits-since/Einsums/Einsums/latest) |
| **Documentation** | [![Documentation](https://img.shields.io/badge/docs-latest-green?style=flat)](https://einsums.github.io/Einsums/) |
| **Connect With Us** | ![Discord](https://img.shields.io/discord/1357368862512906360?logo=discord&label=Discord) |

Provides compile-time contraction pattern analysis to determine optimal operation to perform.

## Requirements
A C++ compiler with C++20 support.

The following libraries are required to build Einsums:

* BLAS and LAPACK
* HDF5

The following libraries are also required, but will be fetched if they can not be found.

* fmtlib >= 11
* Catch2 >= 3
* Einsums/h5cpp
* p-ranav/argparse
* gabime/spdlog >= 1

On my personal development machine, I use MKL for the above requirements. On GitHub Actions, stock BLAS, LAPACK, and FFTW3 are used.

Optional requirements:

* A Fast Fourier Transform library, either FFTW3 or DFT from MKL.
* HIP for graphics card support. Uses hipBlas, hipSolver, and the HIP language. Does not yet support hipFFT.
* cpptrace for backtraces.
* LibreTT for GPU transposes.
* pybind11 for the Python extension module.

## Examples
This will optimize at compile-time to a BLAS dgemm call.
```C++
#include "Einsums/TensorAlgebra.hpp"

using einsums;  // Provides Tensor and create_random_tensor
using einsums::tensor_algebra;  // Provides einsum and Indices
using einsums::index;  // Provides i, j, k

Tensor<2> A = create_random_tensor("A", 7, 7);
Tensor<2> B = create_random_tensor("B", 7, 7);
Tensor<2> C{"C", 7, 7};

einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
```

Two-Electron Contribution to the Fock Matrix
```C++
#include "Einsums/TensorAlgebra.hpp"

using namespace einsums;

void build_Fock_2e_einsum(Tensor<2> *F,
                          const Tensor<4> &g,
                          const Tensor<2> &D) {
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // Will compile-time optimize to BLAS gemv
    einsum(1.0, Indices{p, q}, F,
           2.0, Indices{p, q, r, s}, g, Indices{r, s}, D);

    // As written cannot be optimized.
    // A generic arbitrary contraction function will be used.
    einsum(1.0, Indices{p, q}, F,
          -1.0, Indices{p, r, q, s}, g, Indices{r, s}, D);
}
```

Here are some comparisons between different methods of building the Hartree-Fock G matrix out of the two-electron integrals and the density matrix.
The code for this is similar to the sample above.
The first plot uses timings for 100 ortbitals using several methods: C for loops with compiler loop vectorization; C for loops with
OpenMP loop vectorization and parallelization; Fortran do-concurrent loops; BLAS with a for loop for calculating the K matrix, gemv for the
J matrix, and axpy for the G matrix; BLAS with a for loop to permute the two-electron integrals, then gemv for the J and K matrices
and axpy for the G matrix; Einsums without permuting the two-electron integrals, using the generic algorithm for the K matrix; and
Einsums with a permutation of the two-electron integrals, using a selected algorithm for the K matrix.

![einsum Performance](/docs/sphinx/_static/index-images/Performance.png)

The following shows the difference in overall performance as the number of orbitals increases.

![einsums Growth](/docs/sphinx/_static/index-images/Performance_comp.png)

These timings were computed on a system with  an Intel Core i7-13700K with 32 GB of DDR5 RAM and an
AMD Radeon 7900X graphics card running Debian 12, kernel version 6.1.

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
