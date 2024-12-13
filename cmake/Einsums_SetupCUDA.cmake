#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if (EINSUMS_WITH_CUDA AND NOT TARGET einsums_internal::cuda)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        # nvc++ is used for all source files and we don't enable CMake's CUDA language support as it's
        # not yet supported
        if(NOT EINSUMS_FIND_PACKAGE)
            einsums_add_config_define(EINSUMS_HAVE_CUDA)
        endif()

        add_library(einsums_internal::cuda INTERFACE IMPORTED)

        find_package(CUDAToolkit MODULE REQUIRED)
        target_link_libraries(einsums_internal::cuda INTERFACE CUDA::cudart)
        target_link_libraries(einsums_internal::cuda INTERFACE CUDA::cublas CUDA::cusolver)

        # nvc++ warns about the static keyword coming after the return type. We make use of static
        # coming after the return type in the EINSUMS_STATIC_CALL_OPERATOR macro. Since the order has no
        # significance we outright disable the warning.
        target_compile_options(einsums_internal::cuda INTERFACE --diag_suppress storage_class_not_first)

        if(NOT EINSUMS_FIND_PACKAGE)
            target_link_libraries(einsums_base_libraries INTERFACE einsums_internal::cuda)
            target_compile_options(einsums_private_flags INTERFACE --display_error_number)
        endif()
    else()
        # nvcc or clang are only used for cu files with CMake's CUDA language support
        if(NOT EINSUMS_FIND_PACKAGE)
            if(DEFINED CMAKE_CUDA_STANDARD AND NOT CMAKE_CUDA_STANDARD STREQUAL EINSUMS_WITH_CXX_STANDARD)
                einsums_error(
                        "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD} and EINSUMS_WITH_CXX_STANDARD to ${EINSUMS_WITH_CXX_STANDARD}. Please unset CMAKE_CUDA_STANDARD."
                )
            endif()

            set(CMAKE_CUDA_STANDARD ${EINSUMS_WITH_CXX_STANDARD})
        endif()

        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

        enable_language(CUDA)

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            set(EINSUMS_WITH_CLANG_CUDA ON)
        endif()

        if(NOT EINSUMS_FIND_PACKAGE)
            einsums_add_config_define(EINSUMS_HAVE_CUDA)
        endif()

        # CUDA libraries used
        add_library(einsums_internal::cuda INTERFACE IMPORTED)
        # Toolkit targets like CUDA::cudart, CUDA::cublas, CUDA::cufft, etc. available
        find_package(CUDAToolkit MODULE REQUIRED)
        if(CUDAToolkit_FOUND)
            target_link_libraries(einsums_internal::cuda INTERFACE CUDA::cudart)
            target_link_libraries(einsums_internal::cuda INTERFACE CUDA::cublas CUDA::cusolver)
        endif()
        # Flag not working for CLANG CUDA
        target_compile_features(einsums_internal::cuda INTERFACE cuda_std_${EINSUMS_WITH_CXX_STANDARD})
        set_target_properties(einsums_internal::cuda PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)

        target_compile_definitions(
                einsums_internal::cuda
                INTERFACE
                $<$<AND:$<CUDA_COMPILER_ID:Clang>,$<COMPILE_LANGUAGE:CUDA>>:FMT_USE_FLOAT128=0>
                $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<COMPILE_LANGUAGE:CUDA>>:FMT_USE_NONTYPE_TEMPLATE_ARGS=0>
        )
        if(EINSUMS_WITH_CXX_STANDARD GREATER_EQUAL 20)
            target_compile_definitions(
                    einsums_internal::cuda
                    INTERFACE $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<COMPILE_LANGUAGE:CUDA>>:FMT_USE_CONSTEXPR=1>
            )
        endif()

        if(EINSUMS_WITH_CLANG_CUDA)
            if(NOT EINSUMS_FIND_PACKAGE)
                einsums_add_target_compile_option(-DBOOST_THREAD_USES_MOVE PUBLIC)
            endif()
        else()
            if(MSVC)
                set(CUDA_PROPAGATE_HOST_FLAGS OFF)
                target_compile_options(
                        einsums_internal::cuda
                        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:
                        -D_DEBUG
                        -O0
                        -g
                        -G
                        -Xcompiler=-MDd
                        -Xcompiler=-Od
                        -Xcompiler=-Zi
                        -Xcompiler=-bigobj
                        >>
                )
                target_compile_options(
                        einsums_internal::cuda
                        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:RelWithDebInfo>:
                        -DNDEBUG
                        -O2
                        -g
                        -Xcompiler=-MD,-O2,-Zi
                        -Xcompiler=-bigobj
                        >>
                )
                target_compile_options(
                        einsums_internal::cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:MinSizeRel>: -DNDEBUG
                        -O1 -Xcompiler=-MD,-O1 -Xcompiler=-bigobj >>
                )
                target_compile_options(
                        einsums_internal::cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>: -DNDEBUG
                        -O2 -Xcompiler=-MD,-Ox -Xcompiler=-bigobj >>
                )
            endif()
            set(CUDA_SEPARABLE_COMPILATION ON)
            target_compile_options(
                    einsums_internal::cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda
                    --default-stream per-thread --expt-relaxed-constexpr >
            )
            if(NOT EINSUMS_FIND_PACKAGE)
                if(EINSUMS_WITH_COMPILER_WARNINGS_AS_ERRORS)
                    target_compile_options(
                            einsums_private_flags
                            INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--Werror
                            all-warnings>
                    )
                endif()

                target_compile_options(
                        einsums_private_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--display-error-number>
                )
            endif()
        endif()

        if(NOT EINSUMS_FIND_PACKAGE)
            target_link_libraries(einsums_base_libraries INTERFACE einsums_internal::cuda)
        endif()
    endif()
endif()

function(einsums_add_nvhpc_cuda_flags source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE CXX)

    get_source_file_property(source_compile_flags ${source} COMPILE_FLAGS)
    if(source_compile_flags STREQUAL "NOTFOUND")
        set(source_compile_flags)
    endif()

    set(source_gpu_cc_flags)
    foreach(cuda_arch ${CMAKE_CUDA_ARCHITECTURES})
        set(source_gpu_cc_flags "${source_gpu_cc_flags} -gpu=cc${cuda_arch}")
    endforeach()

    set_source_files_properties(
            ${source} PROPERTIES COMPILE_FLAGS "${source_compile_flags} -x cu ${source_gpu_cc_flags}"
    )
endfunction()
