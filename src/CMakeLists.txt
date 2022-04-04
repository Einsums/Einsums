add_library(einsums

    backends/netlib/dgemm.cpp
    backends/netlib/lsame.cpp
    backends/netlib/xerbla.cpp

    backends/vendor/Vendor.cpp

    Blas.cpp
    Memory.cpp
    Print.cpp
    State.cpp
    Timer.cpp
    $<$<NOT:$<TARGET_EXISTS:OpenMP::OpenMP_CXX>>:OpenMP.c>
)

if (EINSUMS_USE_HPTT)
    target_compile_definitions(einsums PUBLIC EINSUMS_USE_HPTT)
endif()

target_link_libraries(einsums
    PUBLIC
        project-options
        LAPACK::LAPACK
        BLAS::BLAS
        range-v3::range-v3
        fmt::fmt
        h5cpp::h5cpp
        $<$<BOOL:EINSUMS_USE_HPTT>:hptt::hptt>
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
        Backward::Backward
)

target_include_directories(einsums
    PUBLIC
        include
)