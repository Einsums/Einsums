#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_extend_with_python_headers target)
  target_link_libraries(${target} PRIVATE pybind11::headers)
  target_include_directories(${target} PRIVATE ${Python_INCLUDE_DIRS})
endfunction()

function(einsums_extend_with_python target)
  # if (APPLE) target_link_options(${target} PUBLIC -undefined dynamic_lookup) endif ()

  # target_include_directories(${target} PRIVATE ${Python_INCLUDE_DIRS}) target_link_libraries(
  # ${target} PRIVATE pybind11::module pybind11::lto pybind11::windows_extras )

  target_link_libraries(${target} PRIVATE pybind11::module pybind11::lto pybind11::embed)
  einsums_extend_with_python_headers(${target})

  if(MSVC)
    target_link_libraries(${target} PRIVATE pybind11::windows_extras)
  endif()

  pybind11_extension(${target})

  if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${target})
  endif()

  if(APPLE)
    set_target_properties(
      ${target}
      PROPERTIES PREFIX ""
               DEBUG_POSTFIX ""
               INSTALL_RPATH "${CMAKE_INSTALL_RPATH};\@loader_path/../"
    )
  elseif(NOT MSVC)
    set_target_properties(
      ${target}
      PROPERTIES PREFIX ""
               DEBUG_POSTFIX ""
               INSTALL_RPATH "${CMAKE_INSTALL_RPATH};\$ORIGIN/../"
    )
  else()
    set_target_properties(
      ${target}
      PROPERTIES PREFIX ""
                  DEBUG_POSTFIX ""
    )
  endif()
endfunction()
