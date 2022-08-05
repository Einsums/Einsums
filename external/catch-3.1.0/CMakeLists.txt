add_library(Catch2 STATIC catch.cpp)
target_include_directories(Catch2 PUBLIC include)
target_compile_definitions(Catch2 PRIVATE CATCH_AMALGAMATED_CUSTOM_MAIN)

if (UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES Intel OR CMAKE_CXX_COMPILIER_ID MATCHES IntelLLVM))
    target_compile_options(Catch2 PRIVATE -fp-model=precise)
endif()

add_library(Catch2::Catch2 ALIAS Catch2)
