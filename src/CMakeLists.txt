add_einsums_library(einsums
    SOURCES
        backends/netlib/dgemm.cpp
        backends/netlib/lsame.cpp
        backends/netlib/xerbla.cpp

        backends/vendor/Vendor.cpp

        Blas.cpp
        Memory.cpp
        Print.cpp
        Section.cpp
        State.cpp
        Timer.cpp
    PUBLIC_INCLUDES
        include
    DEPENDS
        LAPACK::LAPACK
        BLAS::BLAS
        Backward::Backward
    PUBLIC_DEPENDS
        range-v3::range-v3
        fmt::fmt
        h5cpp::h5cpp
)

extend_einsums_target(einsums
    SOURCES
        backends/onemkl/onemkl.cpp
    DEPENDS
        Intel::SYCL
    CONDITION
        TARGET Intel::SYCL
)

extend_einsums_target(einsums
    SOURCES
        OpenMP.c
    CONDITION
        NOT TARGET OpenMP::OpenMP_CXX
)

extend_einsums_target(einsums
    PUBLIC_DEPENDS
        OpenMP::OpenMP_CXX
    CONDITION
        TARGET OpenMP::OpenMP_CXX
)

extend_einsums_target(einsums
    SOURCES
        backends/cblas/cblas.cpp
    DEFINES
        EINSUMS_HAVE_LAPACKE
    DEPENDS
        lapacke
        LAPACK::LAPACK
    CONDITION
        HAVE_LAPACKE_HEADER
)

extend_einsums_target(einsums
    SOURCES
        backends/cblas/cblas.cpp
    DEFINES
        EINSUMS_HAVE_MKL_LAPACKE
    CONDITION
        HAVE_MKL_LAPACKE_HEADER
)

extend_einsums_target(einsums
    PUBLIC_DEFINES
        EINSUMS_USE_HPTT
    PUBLIC_DEPENDS
        hptt::hptt
    CONDITION
        EINSUMS_USE_HPTT
)

extend_einsums_target(einsums
    DEPENDS
        ittnotify
    DEFINES
        HAVE_ITTNOTIFY
    CONDITION
        TARGET ittnotify
)

extend_einsums_target(einsums
    PUBLIC_OPTIONS
        -fp-model=precise
    CONDITION
        (UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES Intel OR CMAKE_CXX_COMPILIER_ID MATCHES IntelLLVM))
)

list(APPEND defines
    EINSUMS_CONTINUOUSLY_TEST_EINSUM
    EINSUMS_TEST_EINSUM_ABORT
    EINSUMS_WARN_NOT_IMPLEMENTED
    EINSUMS_SHOW_WARNING
    EINSUMS_TEST_NANS
    EINSUMS_RUNTIME_INDICES_CHECK
)
foreach(define IN LISTS defines)
    extend_einsums_target(einsums
        DEFINES
            ${define}
        CONDITION
            ${define}
    )
endforeach()