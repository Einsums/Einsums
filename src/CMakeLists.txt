add_library(EinsumsInCpp
    Blas.cpp
    Memory.cpp
    Print.cpp
    State.cpp
    Timer.cpp
    $<$<NOT:$<BOOL:OpenMP_FOUND>>:OpenMP.c>
)

if (EINSUMS_USE_HPTT)
target_compile_definitions(EinsumsInCpp PUBLIC EINSUMS_USE_HPTT)
endif()

target_link_libraries(EinsumsInCpp
    PUBLIC
        project-options
        LAPACK::LAPACK
        BLAS::BLAS
        range-v3::range-v3
        fmt::fmt
        h5cpp::h5cpp
        $<$<BOOL:EINSUMS_USE_HPTT>:hptt::hptt>
        $<$<BOOL:OpenMP_FOUND>:OpenMP::OpenMP_CXX>
)

target_include_directories(EinsumsInCpp
    PUBLIC
        include
)