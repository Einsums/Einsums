//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>
#include <Einsums/TypeSupport/GPUCast.hpp>
#include <Einsums/TypeSupport/GPUComplex.hpp>
#include <Einsums/hipBLAS.hpp>

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <hipsolver/internal/hipsolver-types.h>

namespace einsums {
namespace linear_algebra {
namespace detail {

namespace gpu {

template <size_t Rank, typename T1, typename T2>
__global__ void dot_kernel(BiggestTypeT<T1, T2> *C, T1 const *__restrict__ A, T2 const *__restrict__ B, size_t const *__restrict__ dims,
                           size_t const *__restrict__ strides, size_t const *__restrict__ A_strides, size_t const *__restrict__ B_strides) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    BiggestTypeT<T1, T2> temp;
    make_zero(temp);

    size_t A_index, B_index;

    for (size_t curr_index = thread_id; curr_index < dims[0] * strides[0]; curr_index += kernel_size) {
        A_index         = 0;
        B_index         = 0;
        size_t quotient = curr_index;

#pragma unroll
        for (int i = 0; i < Rank; i++) {
            A_index += (quotient / strides[i]) * A_strides[i];
            B_index += (quotient / strides[i]) * B_strides[i];
            quotient %= strides[i];
        }

        temp = einsums::gpu_ops::fma(A[A_index], B[B_index], temp);
    }

    einsums::gpu::atomicAdd_wrap(C, temp);
}

template <size_t Rank, typename T1, typename T2>
__global__ void true_dot_kernel(BiggestTypeT<T1, T2> *C, T1 const *__restrict__ A, T2 const *__restrict__ B,
                                size_t const *__restrict__ dims, size_t const *__restrict__ strides, size_t const *__restrict__ A_strides,
                                size_t const *__restrict__ B_strides) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    BiggestTypeT<T1, T2> temp;
    make_zero(temp);

    size_t A_index, B_index;

    for (size_t curr_index = thread_id; curr_index < dims[0] * strides[0]; curr_index += kernel_size) {
        A_index         = 0;
        B_index         = 0;
        size_t quotient = curr_index;

        for (int i = 0; i < Rank; i++) {
            A_index += (quotient / strides[i]) * A_strides[i];
            B_index += (quotient / strides[i]) * B_strides[i];
            quotient %= strides[i];
        }

        if constexpr (std::is_same_v<T1, hipFloatComplex> || std::is_same_v<T1, hipDoubleComplex>) {
            T1 conjugate = A[curr_index];
            conjugate.y  = -conjugate.y;
            temp         = einsums::gpu_ops::fma(conjugate, B[curr_index], temp);
        } else {
            temp = einsums::gpu_ops::fma(A[curr_index], B[curr_index], temp);
        }
    }

    einsums::gpu::atomicAdd_wrap(C, temp);
}

/**
 * Symmetric multiplication kernels.
 */
__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, float const *A, int lda, float const *B, int ldb, float *C,
                                         int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, double const *A, int lda, double const *B, int ldb,
                                         double *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipFloatComplex const *A, int lda,
                                         hipFloatComplex const *B, int ldb, hipFloatComplex *C, int ldc);

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipDoubleComplex const *A, int lda,
                                         hipDoubleComplex const *B, int ldb, hipDoubleComplex *C, int ldc);

} // namespace gpu

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::is_same_v<typename AType::host_datatype, T>;
    }
void gemm(char transA, char transB, T alpha, AType const &A, BType const &B, T beta, CType *C) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    int m = C->dim(0), n = C->dim(1), k = (std::tolower(transA) != 'n') ? A.dim(0) : A.dim(1);
    int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    // Flip the A and B matrices. Row-major vs column major.
    char new_transA = (std::tolower(transA) == 'n')? 't': 'n', new_transB = (std::tolower(transB) == 'n')? 't': 'n';
    blas::gpu::gemm(transB, transA, n, m, k, alpha, (typename AType::ValueType *)B.gpu_data(), ldb,
                    (typename AType::ValueType *)A.gpu_data(), lda, beta, (typename AType::ValueType *)C->gpu_data(), ldc);
    stream_wait();
}

template <bool TransA, bool TransB, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType,
          typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::is_same_v<typename AType::host_datatype, T>;
    }
void gemm(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    gemm((TransA) ? 't' : 'n', (TransB) ? 't' : 'n', alpha, A, B, beta, C);
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType>
    requires(SameRank<AType, BType>)
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    using namespace einsums::gpu;

    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;
    using dev_datatype    = typename tensor_base::DeviceTypedTensor<T>::dev_datatype;
    constexpr size_t Rank = AType::Rank;

    bool is_A_vectorable = true, is_B_vectorable = true;

    if constexpr (AType::Rank != 1) {
        size_t A_check = A.stride(-1), B_check = B.stride(-1);

        for (int i = Rank - 1; i >= 0; i--) {
            if (A.stride(i) != A_check) {
                is_A_vectorable = false;
                break;
            }

            if (B.stride(i) != B_check) {
                is_B_vectorable = false;
                break;
            }

            A_check *= A.dim(i);
            B_check *= B.dim(i);
        }
    }

    if (is_A_vectorable && is_B_vectorable) {
        return blas::gpu::dot(A.size(), (typename AType::ValueType *)A.gpu_data(), A.stride(-1), (typename AType::ValueType *)B.gpu_data(),
                              B.stride(-1));
    } else {

        dev_datatype            *gpu_out;
        auto                     grid       = block_size(A.size());
        auto                     num_blocks = blocks(A.size());
        size_t                  *gpu_strides;
        std::array<size_t, Rank> strides;

        hip_catch(hipMalloc((void **)&gpu_out, sizeof(T)));
        hip_catch(hipMalloc((void **)&gpu_strides, Rank * sizeof(size_t)));

        size_t prod = 1;

        for (int i = Rank - 1; i >= 0; i--) {
            strides[i] = prod;
            prod *= A.dim(i);
        }

        hip_catch(hipMemcpy(gpu_strides, strides.data(), Rank * sizeof(size_t), hipMemcpyHostToDevice));
        // Zero the initial value.
        T out{0.0};
        hip_catch(hipMemcpy(gpu_out, &out, sizeof(T), hipMemcpyHostToDevice));

        gpu::dot_kernel<Rank><<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(
            gpu_out, A.gpu_data(), B.gpu_data(), A.gpu_dims(), gpu_strides, A.gpu_strides(), B.gpu_strides());
        stream_wait();

        hip_catch(hipMemcpy((void *)&out, (void *)gpu_out, sizeof(T), hipMemcpyDeviceToHost));
        // No sync

        hip_catch(hipFree((void *)gpu_out));
        hip_catch(hipFree((void *)gpu_strides));

        return out;
    }
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType>
    requires(SameRank<AType, BType>)
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    using namespace einsums::gpu;

    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;
    using dev_datatype    = typename tensor_base::DeviceTypedTensor<T>::dev_datatype;
    constexpr size_t Rank = AType::Rank;

    bool is_A_vectorable = true, is_B_vectorable = true;

    if constexpr (AType::Rank != 1) {
        size_t A_check = A.stride(-1), B_check = B.stride(-1);

        for (int i = Rank - 1; i >= 0; i--) {
            if (A.stride(i) != A_check) {
                is_A_vectorable = false;
                break;
            }

            if (B.stride(i) != B_check) {
                is_B_vectorable = false;
                break;
            }

            A_check *= A.dim(i);
            B_check *= B.dim(i);
        }
    }

    if (is_A_vectorable && is_B_vectorable) {
        return blas::gpu::dot(A.size(), (typename AType::ValueType *)A.gpu_data(), A.stride(-1), (typename AType::ValueType *)B.gpu_data(),
                              B.stride(-1));
    } else {

        dev_datatype            *gpu_out;
        size_t                  *gpu_strides;
        std::array<size_t, Rank> strides;

        auto grid       = block_size(A.size());
        auto num_blocks = blocks(A.size());

        hip_catch(hipMalloc((void **)&gpu_out, sizeof(T)));
        hip_catch(hipMalloc((void **)&gpu_strides, Rank * sizeof(size_t)));

        size_t prod = 1;

        for (int i = Rank - 1; i >= 0; i--) {
            strides[i] = prod;
            prod *= A.dim(i);
        }

        hip_catch(hipMemcpy(gpu_strides, strides.data(), Rank * sizeof(size_t), hipMemcpyHostToDevice));

        // Zero the initial value
        T out{0.0};
        hip_catch(hipMemcpy(gpu_out, &out, sizeof(T), hipMemcpyHostToDevice));

        gpu::true_dot_kernel<Rank><<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(
            gpu_out, A.gpu_data(), B.gpu_data(), A.gpu_dims(), gpu_strides, A.gpu_strides(), B.gpu_strides());
        stream_wait();

        hip_catch(hipMemcpy((void *)&out, (void *)gpu_out, sizeof(T), hipMemcpyDeviceToHost));
        // No sync

        hip_catch(hipFree((void *)gpu_out));
        hip_catch(hipFree((void *)gpu_strides));

        return out;
    }
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void ger(T alpha, XType const &X, YType const &Y, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    blas::gpu::ger(Y.dim(0), X.dim(0), alpha, (typename AType::ValueType *)Y.gpu_data(), Y.stride(0),
                   (typename AType::ValueType *)X.gpu_data(), X.stride(0), (typename AType::ValueType *)A->gpu_data(), A->stride(0));

    // No wait needed. sort waits.
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void gerc(T alpha, XType const &X, YType const &Y, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    DeviceTensor<T, 1> X_temp("x_temp", X.dim(0));

    X_temp = X;

    blas::gpu::lacgv(X_temp.dim(0), (T *)X_temp.gpu_data(), X_temp.stride(0));

    blas::gpu::ger(Y.dim(0), X.dim(0), alpha, (typename AType::ValueType *)Y.gpu_data(), Y.stride(0),
                    (typename AType::ValueType *)X.gpu_data(), X.stride(0), (typename AType::ValueType *)A->gpu_data(), A->stride(0));

    // No wait needed. sort waits.
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void gemv(char transA, T alpha, AType const &A, XType const &x, T beta, YType *y) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    int m = A.dim(1), n = A.dim(0);

    if (!strchr("CNTcnt", transA)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Invalid transpose parameter! Expected c, n, or t case insensitive. Got {}.",
                                transA);
    }

    if (std::tolower(transA) == 'n') {
        blas::gpu::gemv('t', m, n, alpha, (typename AType::ValueType *)A.gpu_data(), A.stride(0), (typename AType::ValueType *)x.gpu_data(),
                        x.stride(0), beta, (typename AType::ValueType *)y->gpu_data(), y->stride(0));
    } else if (std::tolower(transA) == 't') {
        blas::gpu::gemv('n', m, n, alpha, (typename AType::ValueType *)A.gpu_data(), A.stride(0), (typename AType::ValueType *)x.gpu_data(),
                        x.stride(0), beta, (typename AType::ValueType *)y->gpu_data(), y->stride(0));
    } else {
        if constexpr (IsComplexV<T>) {
            DeviceTensor<T, 1> x_temp("temp", x.dim(0));

            x_temp = x;

            blas::gpu::lacgv(x_temp.dim(0), (typename AType::ValueType *)x_temp.gpu_data(), x_temp.stride(0));
            blas::gpu::lacgv(y->dim(0), (typename AType::ValueType *)y->gpu_data(), y->stride(0));

            blas::gpu::gemv('n', m, n, std::conj(alpha), (typename AType::ValueType *)A.gpu_data(), A.stride(0),
                            (typename AType::ValueType *)x_temp.gpu_data(), x_temp.stride(0), std::conj(beta),
                            (typename AType::ValueType *)y->gpu_data(), y->stride(0));

            blas::gpu::lacgv(y->dim(0), (typename AType::ValueType *)y->gpu_data(), y->stride(0));
        } else {
            blas::gpu::gemv('n', m, n, alpha, (typename AType::ValueType *)A.gpu_data(), A.stride(0),
                            (typename AType::ValueType *)x.gpu_data(), x.stride(0), beta, (typename AType::ValueType *)y->gpu_data(),
                            y->stride(0));
        }
    }

    stream_wait();
}

template <bool TransA, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XType>;
        requires VectorConcept<YType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void gemv(T alpha, AType const &A, XType const &x, T beta, YType *y) {
    gemv((TransA) ? 't' : 'n', alpha, A, x, beta, y);
}

template <DeviceBasicTensorConcept AType, typename T>
    requires(std::is_same_v<typename AType::ValueType, T>)
void scale(T alpha, AType *A) {
    using namespace einsums::gpu;

    using dev_datatype = typename AType::dev_datatype;

    /// @todo Compatibility with views.
    blas::gpu::scal(A->size(), alpha, (typename AType::ValueType *)A->gpu_data(), 1);
    stream_wait();
}

template <bool TransA, bool TransB, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
    using namespace einsums::gpu;

    if constexpr (TransA && TransB) {
        assert(B.dim(0) == A.dim(0) && A.dim(1) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else if constexpr (TransA && !TransB) {
        assert(B.dim(1) == A.dim(0) && A.dim(1) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    } else if constexpr (!TransA && TransB) {
        assert(B.dim(0) == A.dim(1) && A.dim(0) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
    } else {
        assert(B.dim(1) == A.dim(1) && A.dim(0) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
    }
    *C = typename CType::ValueType(0.0);

    einsums::linear_algebra::detail::gpu::symm_gemm<<<block_size(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)),
                                                      blocks(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), 0, get_stream()>>>(
        TransA, TransB, A.dim(0), C->dim(0), A.gpu_data(), A.stride(0), B.gpu_data(), B.stride(0), C->gpu_data(), C->stride(0));
    stream_wait();
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType>;
    }
int gesv(AType *A, BType *B) {
    using namespace einsums::gpu;
    auto n   = A->dim(0);
    auto lda = A->stride(0);
    auto ldb = B->stride(0);

    auto nrhs = B->dim(0);

    int lwork = n;
    int info;

    int *ipiv;

#ifdef __HIP_PLATFORM_NVIDIA__
    DeviceTensor<typename AType::ValueType, 2> X = DeviceTensor<typename AType::ValueType, 2>(einsums::detail::DEV_ONLY, B->dims());
#endif

    einsums::hip_catch(hipMallocAsync((void **)&ipiv, sizeof(int) * lwork, einsums::gpu::get_stream()));

#ifdef __HIP_PLATFORM_NVIDIA__
    info = blas::gpu::gesv(n, nrhs, (typename AType::ValueType *)A->gpu_data(), lda, ipiv, (typename AType::ValueType *)B->gpu_data(), ldb,
                           (typename AType::ValueType *)X.gpu_data(), X.stride(0));
#elif defined(__HIP_PLATFORM_AMD__)
    info = blas::gpu::gesv(n, nrhs, (typename AType::ValueType *)A->gpu_data(), lda, ipiv, (typename AType::ValueType *)B->gpu_data(), ldb,
                           (typename AType::ValueType *)B->gpu_data(), ldb);
#endif

    stream_wait();

    einsums::hip_catch(hipFreeAsync(ipiv, einsums::gpu::get_stream()));
    return info;
}

template <DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<XType, YType>;
        requires std::is_same_v<typename XType::ValueType, T>;
    }
void axpy(T alpha, XType const &X, YType *Y) {
    using namespace einsums::gpu;

    blas::gpu::axpy(X.dim(0) * X.stride(0), alpha, (typename XType::ValueType *)X.gpu_data(), 1, (typename XType::ValueType *)Y->gpu_data(),
                    1);
    stream_wait();
}

template <DeviceBasicTensorConcept XType, DeviceBasicTensorConcept YType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<XType, YType>;
        requires std::is_same_v<typename XType::ValueType, T>;
    }
void axpby(T alpha, XType const &X, T beta, YType *Y) {
    using namespace einsums::gpu;

    blas::gpu::axpby(X.dim(0) * X.stride(0), alpha, (typename XType::ValueType *)X.gpu_data(), 1, beta,
                     (typename XType::ValueType *)Y->gpu_data(), 1);
    stream_wait();
}

/**
 * @brief Computes the eigenvalues and eigenvectors of a symmetric matrix.
 */

template <bool ComputeEigenvectors = true, DeviceBasicTensorConcept AType, DeviceBasicTensorConcept WType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<WType>;
        requires SameUnderlying<AType, WType>;
    }
void syev(AType *A, WType *W) {
    using namespace einsums::gpu;
    int lda = A->stride(0);

    int info = blas::gpu::syev(ComputeEigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER,
                               A->dim(0), (typename AType::ValueType *)A->gpu_data(), lda, (typename WType::ValueType *)W->gpu_data());

    stream_wait();
}

template <DeviceBasicTensorConcept AType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void scale_row(size_t row, T alpha, AType *A) {
    using namespace einsums::gpu;
    blas::gpu::scal(A->dim(1), alpha, (typename AType::ValueType *)A->gpu_data(row, 0ul), A->stride(1));
    stream_wait();
}

template <DeviceBasicTensorConcept AType, typename T>
    requires requires {
        requires MatrixConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void scale_column(size_t col, T alpha, AType *A) {
    using namespace einsums::gpu;
    blas::gpu::scal(A->dim(0), alpha, (typename AType::ValueType *)A->gpu_data(0ul, col), A->stride(0));
    stream_wait();
}

template <DeviceBasicTensorConcept AType, DeviceBasicTensorConcept BType, DeviceBasicTensorConcept CType, typename T>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
void direct_product(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    using namespace einsums::gpu;

    using T_devtype  = typename CType::dev_datatype;
    using T_hosttype = typename CType::host_datatype;

    constexpr size_t Rank = CType::Rank;

    assert(A.dims() == B.dims() && A.dims() == C->dims());

    size_t elems = A.size();

    scale(beta, C);

    blas::gpu::dirprod(elems, alpha, (typename AType::ValueType *)A.gpu_data(), A.stride(-1), (typename AType::ValueType *)B.gpu_data(),
                       B.stride(-1), (typename AType::ValueType *)C->gpu_data(), C->stride(-1));
}

namespace gpu {
/**
 * @brief Copy a list of eigenvalues onto the diagonal of a matrix.
 *
 * @param out_matrix The matrix output. Only the diagonal entries are touched.
 * @param n The number of columns and the number of eigenvalues.
 * @param lda The leading dimension of the out_matrix. lda >= n.
 * @param eigs The eigenvalues to copy.
 */
template <typename T, typename U>
__global__ void eig_to_diag(T *__restrict__ out_matrix, int n, int lda, U const *__restrict__ eigs, T expo) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    // Copy to diagonal. Assume the matrix is zeroed, or at least that the user needs the off-diagonal entries.
    for (int i = thread_id; i < n; i += num_threads) {
        out_matrix[i * lda + i] = ::pow(HipCast<T, U>::cast(eigs[i]), expo);
    }
}
} // namespace gpu

template <MatrixConcept AType>
    requires DeviceBasicTensorConcept<AType>
auto pow(AType const &a, typename AType::host_datatype alpha,
         typename AType::host_datatype cutoff = std::numeric_limits<typename AType::host_datatype>::epsilon()) -> RemoveViewT<AType> {
    using T = typename AType::host_datatype;
    DeviceTensor<RemoveComplexT<T>, 1> Evals(Dim<1>{a.dim(0)}, ::einsums::detail::DEV_ONLY);

    RemoveViewT<AType> Evecs = create_tensor_like(a);

    RemoveViewT<AType> Diag = create_tensor_like(a);

    RemoveViewT<AType> out  = create_tensor_like(a);
    RemoveViewT<AType> temp = create_tensor_like(a);

    Evecs.assign(a);

    if constexpr (einsums::IsComplexV<AType>) {
        hyev<true>(&Evecs, &Evals);
    } else {
        syev<true>(&Evecs, &Evals);
    }

    Diag.zero();

    gpu::eig_to_diag<<<dim3(32), dim3(1), 0, einsums::gpu::get_stream()>>>(Diag.gpu_data(), Diag.dim(0), Diag.stride(0), Evals.gpu_data(),
                                                                           alpha);

    symm_gemm<false, false>(Diag, Evecs, &out);

    return out;
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums