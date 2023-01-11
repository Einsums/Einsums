set(_flags "-O1 -g -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer")

foreach(_language C CXX)
    string(REPLACE "X" "+" _human_readable_language ${_language})
    set(CMAKE_${_language}_FLAGS_MSAN ${_flags} CACHE STRING "${_human_readable_language} flags for memory sanitizer" FORCE)
    mark_as_advanced(CMAKE_${_language}_FLAGS_MSAN)
endforeach()