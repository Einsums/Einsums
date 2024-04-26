#pragma once

#include "einsums/_GPUCast.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/GPULinearAlgebra.hpp"

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

template <typename CDataType, typename ADataType, typename BDataType>
__global__ void dot_kernel(CDataType *C, ::std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor,
                           const ADataType *A, const BDataType *B, size_t elements) {
    using namespace einsums::gpu;
    int thread_id, kernel_size;

    get_worker_info(thread_id, kernel_size);

    size_t curr_index = thread_id;

    while (curr_index < elements) {
        if constexpr (std::is_same_v<CDataType, hipComplex> || std::is_same_v<CDataType, hipDoubleComplex>) {
            CDataType term = HipCast<CDataType, decltype(AB_prefactor)>::cast(AB_prefactor * A[curr_index] * B[curr_index]);
            atomicAdd(&(C->x), term.x);
            atomicAdd(&(C->y), term.y);
        } else {
            atomicAdd(C, HipCast<CDataType, decltype(AB_prefactor)>::cast(AB_prefactor * A[curr_index] * B[curr_index]));
        }
        curr_index += kernel_size;
    }
}
} // namespace gpu

} // namespace detail

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<CType<T, Rank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (::einsums::DeviceRankBlockTensor<AType<T, Rank>, 2, T> && ::einsums::DeviceRankBlockTensor<BType<T, Rank>, 2, T>);
    }
void gemm(const T *alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T *beta, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<CType<T, Rank>, Rank, T>) {

        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
            throw std::runtime_error("gemm: Tensors need the same number of blocks.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>(alpha, A.block(i), B.block(i), beta, &(C->block(i)));
        }

        return;
    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                         einsums::detail::IsDeviceRankBlockTensorV<BType<T, Rank>, Rank, T>) {

        if (A.num_blocks() != B.num_blocks()) {
            gemm<TransA, TransB>(alpha, (DeviceTensor<T, 2>)A, (DeviceTensor<T, 2>)B, beta, C);
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (int i = 0; i < A.num_blocks(); i++) {
                if (A.block_dim(i) == 0) {
                    continue;
                }
                gemm<TransA, TransB>(alpha, A.block(i), B.block(i), beta, &((*C)(A.block_range(i), A.block_range(i))));
            }
        }
        return;
    } else {

        using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                                  ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

        int m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
        int lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

        // Flip the A and B matrices. Row-major vs column major.
        detail::gpu::gemm(TransB ? HIPBLAS_OP_T : HIPBLAS_OP_N, TransA ? HIPBLAS_OP_T : HIPBLAS_OP_N, n, m, k, (dev_datatype *)alpha,
                          B.data(), ldb, A.data(), lda, (dev_datatype *)beta, C->data(), ldc);
        stream_wait();
    }
}

template <template <typename, size_t> typename CType, typename CDataType, template <typename, size_t> typename AType, typename ADataType,
          size_t ARank, template <typename, size_t> typename BType, typename BDataType, size_t BRank>
    requires requires {
        requires ::einsums::DeviceRankTensor<CType<CDataType, 0>, 0, CDataType>;
        requires ::einsums::DeviceRankTensor<AType<ADataType, ARank>, ARank, ADataType>;
        requires ::einsums::DeviceRankTensor<BType<BDataType, BRank>, BRank, BDataType>;
    }
void dot(CDataType C_prefactor, CType<CDataType, 0> &C,
         ::std::conditional_t<sizeof(ADataType) < sizeof(BDataType), BDataType, ADataType> AB_prefactor, const AType<ADataType, ARank> &A,
         const BType<BDataType, BRank> &B) {
    using namespace einsums::gpu;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        if (A.num_blocks() != B.num_blocks()) {
            dot(C_prefactor, C, AB_prefactor, (einsums::DeviceTensor<ADataType, ARank>)A, (einsums::DeviceTensor<BDataType, BRank>)B);
        }

        if (A.ranges() != B.ranges()) {
            dot(C_prefactor, C, AB_prefactor, (einsums::DeviceTensor<ADataType, ARank>)A, (einsums::DeviceTensor<BDataType, BRank>)B);
        }

        if (C_prefactor == CDataType{0}) {
            C = CDataType{0};
        } else {
            C *= C_prefactor;
        }

        CDataType out = CDataType{0};

#pragma omp parallel for reduction(+ : out)
        for (int i = 0; i < A.num_blocks(); i++) {

            if (A.block_dim(i) == 0) {
                continue;
            }
            CType<CDataType, 0> temp;
            dot(CDataType{0}, temp, AB_prefactor, A.block(i), B.block(i));

            C += temp;
        }

    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                         !einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        dot(C_prefactor, C, AB_prefactor, (einsums::DeviceTensor<ADataType, ARank>)A, B);
    } else if constexpr (!einsums::detail::IsDeviceRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType> &&
                         einsums::detail::IsDeviceRankBlockTensorV<BType<BDataType, BRank>, BRank, BDataType>) {
        dot(C_prefactor, C, A, AB_prefactor, (einsums::DeviceTensor<BDataType, BRank>)B);

    } else {

        if (C_prefactor == CDataType{0}) {
            C = CDataType{0};
        } else {
            C *= C_prefactor;
        }

        using dev_datatype = std::conditional_t<
            std::is_same_v<decltype(AB_prefactor), std::complex<float>>, hipComplex,
            std::conditional_t<std::is_same_v<decltype(AB_prefactor), std::complex<double>>, hipDoubleComplex, decltype(AB_prefactor)>>;

        detail::gpu::dot_kernel<<<block_size(A.size()), blocks(A.size()), 0, get_stream()>>>(
            C.data(), HipCast<dev_datatype, decltype(AB_prefactor)>::cast(AB_prefactor), A.data(), B.data(), A.size());
        stream_wait();
    }
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<AType<T, ARank>, 2, T>;
    }
void ger(const T *alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    detail::gpu::ger(X.dim(0), Y.dim(0), (dev_datatype *)alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(const T *alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const T *beta, YType<T, XYRank> *y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T>) {

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemv(alpha, A.block(i), x(A.block_range(i)), beta, &((*y)(A.block_range(i))));
        }

    } else {

        int m = A.dim(1), n = A.dim(0);

        if constexpr (!TransA) {
            detail::gpu::gemv(HIPBLAS_OP_T, m, n, (dev_datatype *)alpha, A.data(), A.stride(0), x.data(), x.stride(0), (dev_datatype *)beta,
                              y->data(), y->stride(0));
            stream_wait();
        } else {
            detail::gpu::gemv(HIPBLAS_OP_N, m, n, (dev_datatype *)alpha, A.data(), A.stride(0), x.data(), x.stride(0), (dev_datatype *)beta,
                              y->data(), y->stride(0));
            stream_wait();
        }
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires ::einsums::DeviceRankTensor<AType<T, ARank>, ARank, T>
void scale(const T *scale, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            scale(scale, &(A->block(i)));
        }

        return;
    } else {
        detail::gpu::scal(A->size(), (dev_datatype *)scale, A->data(), 1);
        stream_wait();
    }
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<CType<T, Rank>, 2, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<CType<T, Rank>, Rank, T>) {

        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
            throw std::runtime_error("gemm: Tensors need the same number of blocks.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            symm_gemm<TransA, TransB>(A.block(i), B.block(i), &(C->block(i)));
        }

        return;
    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                         einsums::detail::IsDeviceRankBlockTensorV<BType<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks()) {
            symm_gemm<TransA, TransB>((DeviceTensor<T, 2>)A, (DeviceTensor<T, 2>)B, C);
            stream_wait();
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (int i = 0; i < A.num_blocks(); i++) {
                if (A.block_dim(i) == 0) {
                    continue;
                }
                symm_gemm<TransA, TransB>(A.block(i), B.block(i), &((*C)(A.block_range(i), A.block_range(i))));
            }
        }

        return;
    } else {

        if constexpr (TransA && TransB) {
            assert(B.dim(0) == A.dim(0) && A.dim(1) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
        } else if constexpr (TransA && !TransB) {
            assert(B.dim(1) == A.dim(0) && A.dim(1) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
        } else if constexpr (!TransA && TransB) {
            assert(B.dim(0) == A.dim(1) && A.dim(0) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1));
        } else {
            assert(B.dim(1) == A.dim(1) && A.dim(0) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0));
        }
        C->zero();

        einsums::linear_algebra::detail::gpu::symm_gemm<<<block_size(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)),
                                                          blocks(A.dim(0) * A.dim(0) * C->dim(0) * C->dim(0)), 0, get_stream()>>>(
            TransA, TransB, A.dim(0), C->dim(0), A.data(), A.stride(0), B.data(), B.stride(0), C->data(), C->stride(0));
        stream_wait();
    }
}

template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, Rank>, 2, T>;
        requires ::einsums::DeviceRankTensor<CType<T, Rank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (::einsums::DeviceRankBlockTensor<AType<T, Rank>, 2, T> && ::einsums::DeviceRankBlockTensor<BType<T, Rank>, 2, T>);
        requires(!std::is_pointer_v<T>);
    }
void gemm(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    // Flip the A and B matrices. Row-major vs column major.
    gemm<TransA, TransB>((T *)alpha_gpu, A, B, (T *)beta_gpu, C);

    // These can still be done async.
    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
    hip_catch(hipFreeAsync(beta_gpu, get_stream()));
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires ::einsums::DeviceRankTensor<XYType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires !::einsums::DeviceRankBlockTensor<AType<T, ARank>, 2, T>;
        requires !std::is_pointer_v<T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));

    ger((T *)alpha_gpu, X, Y, A);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<XType<T, XYRank>, 1, T>;
        requires ::einsums::DeviceRankTensor<YType<T, XYRank>, 1, T>;
        requires !std::is_pointer_v<T>;
    }
void gemv(T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, T beta, YType<T, XYRank> *y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;
    dev_datatype *alpha_gpu, *beta_gpu;

    hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
    hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy((void *)alpha_gpu, (const void *)&alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)beta_gpu, (const void *)&beta, sizeof(dev_datatype), hipMemcpyHostToDevice));

    gemv<TransA>((T *)alpha_gpu, A, x, (T *)beta_gpu, y);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
    hip_catch(hipFreeAsync(beta_gpu, get_stream()));
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, ARank, T>;
        requires !std::is_pointer_v<T>;
    }
void scale(T scale_, AType<T, ARank> *A) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *scale_gpu;

    hip_catch(hipMalloc((void **)&scale_gpu, sizeof(dev_datatype)));

    hip_catch(hipMemcpy(scale_gpu, &scale_, sizeof(dev_datatype), hipMemcpyHostToDevice));

    scale((T *)scale_gpu, A);

    hip_catch(hipFreeAsync(scale_gpu, get_stream()));
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<BType<T, BRank>, 2, T>;
    }
int gesv(AType<T, ARank> *A, BType<T, BRank> *B) {
    using namespace einsums::gpu;
    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<BType<T, BRank>, BRank, T>) {

        if (A->num_blocks() != B->num_blocks()) {
            throw std::runtime_error("gesv: Tensors need the same number of blocks.");
        }

        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &(B->block(i)));

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;

    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {

            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &((*B)(AllT(), A->block_range(i))));

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;
    } else {

        LabeledSection0();

        auto n   = A->dim(0);
        auto lda = A->stride(0);
        auto ldb = B->stride(0);

        auto nrhs = B->dim(0);

        int lwork = n;
        int info;

        __device_ptr__ int *ipiv;

#ifdef __HIP_PLATFORM_NVIDIA__
        DeviceTensor<T, BRank> X = DeviceTensor<T, BRank>(einsums::detail::DEV_ONLY, B->dims());
#endif

        einsums::gpu::hip_catch(hipMallocAsync((void **)&ipiv, sizeof(int) * lwork, gpu::get_stream()));

#ifdef __HIP_PLATFORM_NVIDIA__
        info = detail::gpu::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb, X.data(), X.stride(0));
#elif defined(__HIP_PLATFORM_AMD__)
        info = detail::gpu::gesv(n, nrhs, A->data(), lda, ipiv, B->data(), ldb, B->data(), ldb);
#endif

        stream_wait();

        einsums::gpu::hip_catch(hipFreeAsync(ipiv, gpu::get_stream()));

        return info;
    }
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires(!std::is_pointer_v<T>);
        requires DeviceRankTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    using namespace einsums::gpu;

    using dev_datatype = ::std::conditional_t<::std::is_same_v<T, ::std::complex<float>>, hipComplex,
                                              ::std::conditional_t<::std::is_same_v<T, ::std::complex<double>>, hipDoubleComplex, T>>;

    dev_datatype *alpha_gpu;

    hip_catch(hipMallocAsync((void **)&alpha_gpu, sizeof(dev_datatype), get_stream()));

    hip_catch(hipMemcpyAsync(alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice, get_stream()));

    axpy((T *)alpha_gpu, X, Y);

    hip_catch(hipFreeAsync(alpha_gpu, get_stream()));
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires DeviceRankTensor<XType<T, Rank>, Rank, T>;
        requires DeviceRankTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(const T *alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    using namespace einsums::gpu;
    if constexpr (einsums::detail::IsDeviceRankBlockTensorV<XType<T, Rank>, Rank, T> &&
                  einsums::detail::IsDeviceRankBlockTensorV<YType<T, Rank>, Rank, T>) {
        if (X.num_blocks() != Y->num_blocks()) {
            throw std::runtime_error("axpy: Tensors need to have the same number of blocks.");
        }

        if (X.ranges() != Y->ranges()) {
            throw std::runtime_error("axpy: Tensor blocks need to be compatible.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim(i) == 0) {
                continue;
            }

            axpy(alpha, X[i], &(Y->block(i)));
        }
    } else if constexpr (einsums::detail::IsDeviceRankBlockTensorV<XType<T, Rank>, Rank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim(i) == 0) {
                continue;
            }

            std::array<einsums::Range, Rank> slice;

            slice.fill(X.block_range());

            auto Y_block = std::apply(*Y, slice);

            axpy(alpha, X[i], &Y_block);
        }
    } else {

        LabeledSection0();

        detail::gpu::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
        stream_wait();
    }
}

/**
 * @brief Computes the eigenvalues and eigenvectors of a symmetric matrix.
 */
template <bool ComputeEigenvectors, template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType,
          size_t WRank, typename T>
    requires requires {
        requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>;
        requires ::einsums::DeviceRankTensor<WType<T, WRank>, 1, T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {
    using namespace einsums::gpu;
    if constexpr (::einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            syev<ComputeEigenvectors>(&(A->block(i)), &((*W))(A->block_range(i)));
        }
    } else {
        int lda = A->stride(0);

        int info = detail::gpu::syev(ComputeEigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR,
                                     HIPSOLVER_FILL_MODE_UPPER, A->dim(0), A->data(), lda, W->data());

        stream_wait();
    }
}

namespace detail {
namespace gpu {
/**
 * @brief Copy a list of eigenvalues onto the diagonal of a matrix.
 *
 * @param out_matrix The matrix output. Only the diagonal entries are touched.
 * @param n The number of columns and the number of eigenvalues.
 * @param lda The leading dimension of the out_matrix. lda >= n.
 * @param eigs The eigenvalues to copy.
 */
template <typename T>
__global__ void eig_to_diag(T *out_matrix, int n, int lda, const T *eigs, T expo) {
    int thread_id, num_threads;

    einsums::gpu::get_worker_info(thread_id, num_threads);

    // Copy to diagonal. Assume the matrix is zeroed, or at least that the user needs the off-diagonal entries.
    for (int i = thread_id; i < n; i += num_threads) {
        out_matrix[i * lda + i] = ::pow(eigs[i], expo);
    }
}
} // namespace gpu
} // namespace detail

/**
 * @brief Computes the power of a symmetric matrix.
 */
template <template <typename, size_t> typename AType, typename T, size_t ARank>
    requires ::einsums::DeviceRankTensor<AType<T, ARank>, 2, T>
AType<T, ARank> pow(const AType<T, ARank> &A, T expo) {
    using namespace einsums::gpu;
    using namespace einsums;
    if constexpr (::einsums::detail::IsDeviceRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        BlockDeviceTensor<T, 2> out(A);

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < out.num_blocks(); i++) {
            if (out.block_dim(i) == 0) {
                continue;
            }

            out[i] = pow(A.block(i), expo);
        }

        return out;
    } else {
        DeviceTensor<T, 1> Evals(Dim<1>{A.dim(0)}, ::einsums::detail::DEV_ONLY);

        DeviceTensor<T, 2> Evecs(Dim<2>{A.dim(0), A.dim(1)});

        DeviceTensor<T, 2> Diag(Dim<2>{A.dim(0), A.dim(1)}, ::einsums::detail::DEV_ONLY);

        DeviceTensor<T, 2> out(Dim<2>{A.dim(0), A.dim(1)}, ::einsums::detail::DEV_ONLY);
        DeviceTensor<T, 2> temp(Dim<2>{A.dim(0), A.dim(1)}, ::einsums::detail::DEV_ONLY);

        Evecs.assign(A);

        syev(&Evecs, &Evals);

        Diag.zero();

        detail::gpu::eig_to_diag<<<dim3(32), dim3(1), 0, get_stream()>>>(Diag.data(), Diag.dim(0), Diag.stride(0), Evals.data(), expo);

        symm_gemm<false, false>(Diag, Evecs, &out);

        return out;
    }
}

} // namespace linear_algebra
} // namespace einsums