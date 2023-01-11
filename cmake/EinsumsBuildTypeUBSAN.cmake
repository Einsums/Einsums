set(_flags "-O1 -g -fsanitize=undefined -fno-omit-frame-pointer")

foreach(_language C CXX)
    string(REPLACE "X" "+" _human_readable_language ${_language})
    set(CMAKE_${_language}_FLAGS_UBSAN ${_flags} CACHE STRING "${_human_readable_language} flags for undefined behavior sanitizer" FORCE)
    mark_as_advanced(CMAKE_${_language}_FLAGS_UBSAN)
endforeach()