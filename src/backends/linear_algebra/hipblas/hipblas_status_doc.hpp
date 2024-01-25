#pragma once

/**
 * @typedef blasSuccess
 *
 * @brief Represents a success.
 */

/**
 * @typedef blasNotInitialized
 *
 * @brief Indicates that the hipBLAS environment has not been properly initialized.
 *
 * Indicates that the hipBLAS environment has not been properly initialized. Ensure it is initialized by
 * calling @ref einsums::backend::hipblas::initialize() or hipFree(nullptr).
 */

/**
 * @typedef blasAllocFailed
 *
 * @brief Internal error indicating that resource allocation failed.
 * Internal error indicating that resource allocation failed. Ensure there is enough memory for work arrays,
 * and that any work arrays passed have enough memory.
 */

/**
 * @typedef blasInvalidValue
 *
 * @brief Indicates that an unsupported numerical value was passed to a function.
 * Indicates that an unsupported numerical value was passed to a function. An example could be passing an incompatible leading
 * dimension to an array argument.
 */

/**
 * @typedef blasMappingError
 *
 * @brief Indicates that access to the GPU memory space failed.
 * Indicates that access to the GPU memory space failed. This may be caused by passing a host pointer to a GPU function,
 * unmapping or unpinning host memory before calling or during a call, or deallocating a pointer while the GPU is still
 * processing the data.
 */

/**
 * @typedef blasExecutionFailed
 *
 * @brief Indicates that the GPU program failed to execute.
 * Indicates that the GPU program failed to execute. This could have many causes.
 */

/**
 * @typedef blasInternalError
 *
 * @brief Indicates that an unspecified internal error has occurred.
 */

/**
 * @typedef blasNotSupported
 *
 * @brief Indicates that the requested operation is not supported.
 * Indicates that the requested operation is not supported. For instance, calling hipblasXscalBatched with an NVidia card.
 * The batched functions are only supported by AMD cards and rocBLAS, not cuBLAS.
 */

/**
 * @typedef blasArchMismatch
 *
 * @brief Indicates that the code was compiled for a different architecture than is present.
 * Indicates that the code was compiled for a different architecture than is present. Recompile the code for
 * your device to fix this.
 */

 /**
 * @typedef blasHandleIsNullptr
 *
 * @brief Indicates that the handle passed is a null pointer.
 * Indicates that the handle passed is a null pointer. Either set the handle used internally by calling
 * @ref einsums::backend::linear_algebra::hipblas::initialize or by setting it with
 * @ref einsums::backend::linear_algebra::hipblas::set_blas_handle.
 * If you are calling the hipBLAS functions directly, create a new handle using
 * hipblasCreate and use this in subsequent calls.
 */

/**
 * @typedef blasInvalidEnum
 *
 * @brief Indicates that an enum value was passed that was not expected.
 * Indicates that an enum value was passed that was not expected. For instance, passing a value other than
 * HIPBLAS_OP_N (111), HIPBLAS_OP_T (112), or HIPBLAS_OP_C (113) to a transpose argument.
 */

/**
 * @typedef blasUnknown
 *
 * @brief Indicates an unsupported status code.
 * Indicates an unsupported status code was thrown by the backend. It is also thrown if an unsupported
 * status code is passed to @ref einsums::backend::linear_algebra::hipblas::hipblas_catch.
 */


/**
 * @typedef solverSuccess
 *
 * @brief Represents a success.
 */

/**
 * @typedef solverNotInitialized
 *
 * @brief Indicates that the hipSolver environment has not been properly initialized.
 *
 * Indicates that the hipBLAS environment has not been properly initialized. Ensure it is initialized by
 * calling @ref einsums::backend::hipblas::initialize() or hipFree(nullptr).
 */

/**
 * @typedef solverAllocFailed
 *
 * @brief Internal error indicating that resource allocation failed.
 * Internal error indicating that resource allocation failed. Ensure there is enough memory for work arrays,
 * and that any work arrays passed have enough memory.
 */

/**
 * @typedef solverInvalidValue
 *
 * @brief Indicates that an unsupported numerical value was passed to a function.
 * Indicates that an unsupported numerical value was passed to a function. An example could be passing an incompatible leading
 * dimension to an array argument.
 */

/**
 * @typedef solverMappingError
 *
 * @brief Indicates that access to the GPU memory space failed.
 * Indicates that access to the GPU memory space failed. This may be caused by passing a host pointer to a GPU function,
 * unmapping or unpinning host memory before calling or during a call, or deallocating a pointer while the GPU is still
 * processing the data.
 */

/**
 * @typedef solverExecutionFailed
 *
 * @brief Indicates that the GPU program failed to execute.
 * Indicates that the GPU program failed to execute. This could have many causes.
 */

/**
 * @typedef solverInternalError
 *
 * @brief Indicates that an unspecified internal error has occurred.
 */

/**
 * @typedef solverNotSupported
 *
 * @brief Indicates that the requested operation is not supported.
 * Indicates that the requested operation is not supported. For instance, using hipsolverRfBatchSolve on an AMD
 * device, which is currently not supported.
 */

/**
 * @typedef solverArchMismatch
 *
 * @brief Indicates that the code was compiled for a different architecture than is present.
 * Indicates that the code was compiled for a different architecture than is present. Recompile the code for
 * your device to fix this.
 */

 /**
 * @typedef solverHandleIsNullptr
 *
 * @brief Indicates that the handle passed is a null pointer.
 * Indicates that the handle passed is a null pointer. Either set the handle used internally by calling
 * @ref einsums::backend::linear_algebra::hipblas::initialize or by setting it with
 * @ref einsums::backend::linear_algebra::hipblas::set_solver_handle.
 * If you are calling the hipSolver functions directly, create a new handle using
 * hipsolverCreate and use this in subsequent calls.
 */

/**
 * @typedef solverInvalidEnum
 *
 * @brief Indicates that an enum value was passed that was not expected.
 * Indicates that an enum value was passed that was not expected. For instance, passing a value other than
 * HIPSOLVER_OP_N (111), HIPSOLVER_OP_T (112), or HIPSOLVER_OP_C (113) to a transpose argument.
 */

/**
 * @typedef solverUnknown
 *
 * @brief Indicates an unsupported status code.
 * Indicates an unsupported status code was thrown by the backend. It is also thrown if an unsupported
 * status code is passed to @ref einsums::backend::linear_algebra::hipblas::hipsolver_catch.
 */