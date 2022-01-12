add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE include)
add_library(Catch2::Catch2 ALIAS Catch2)