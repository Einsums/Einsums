set(_flags "-O1 -g -fsanitize=address -fno-omit-frame-pointer")

foreach(_language C CXX)
    string(REPLACE "X" "+" _human_readable_language ${_language})
    set(CMAKE_${_language}_FLAGS_ASAN ${_flags} CACHE STRING "${_human_readable_language} flags for address sanitizer" FORCE)
    mark_as_advanced(CMAKE_${_language}_FLAGS_ASAN)
endforeach()