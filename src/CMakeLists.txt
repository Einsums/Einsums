add_library(einsums

    backends/netlib/dgemm.cpp
    backends/netlib/lsame.cpp
    backends/netlib/xerbla.cpp

    backends/vendor/Vendor.cpp

    $<$<TARGET_EXISTS:Intel::SYCL>:backends/onemkl/onemkl.cpp>

    Blas.cpp
    Memory.cpp
    Print.cpp
    Section.cpp
    State.cpp
    Timer.cpp
    $<$<NOT:$<TARGET_EXISTS:OpenMP::OpenMP_CXX>>:OpenMP.c>
)

if (HAVE_LAPACKE_HEADER)
    target_sources(einsums PRIVATE backends/cblas/cblas.cpp)
    target_compile_definitions(einsums PRIVATE EINSUMS_HAVE_LAPACKE)
endif()

if (HAVE_MKL_LAPACKE_HEADER)
    target_sources(einsums PRIVATE backends/cblas/cblas.cpp)
    target_compile_definitions(einsums PRIVATE EINSUMS_HAVE_MKL_LAPACKE)
endif()

if (EINSUMS_USE_HPTT)
    target_compile_definitions(einsums PUBLIC EINSUMS_USE_HPTT)
endif()

target_link_libraries(einsums
    PUBLIC
        einsums-project-options
        $<$<TARGET_EXISTS:lapacke>:lapacke>
        LAPACK::LAPACK
        BLAS::BLAS
        range-v3::range-v3
        fmt::fmt
        h5cpp::h5cpp
        $<$<TARGET_EXISTS:hptt::hptt>:hptt::hptt>
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
        $<$<TARGET_EXISTS:Intel::SYCL>:Intel::SYCL>
        Backward::Backward

    PRIVATE
        $<$<TARGET_EXISTS:ittnotify>:ittnotify>
)

target_include_directories(einsums
    PUBLIC
        include
)