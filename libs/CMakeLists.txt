#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_Message)
include(Einsums_AddPseudoDependencies)
include(Einsums_AddPseudoTarget)

# add example pseudo targets needed for modules
if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(examples.modules)
  einsums_add_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
if(EINSUMS_WITH_TESTS)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(tests.unit.modules)
    einsums_add_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if(EINSUMS_WITH_EXAMPLES AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(tests.examples.modules)
    einsums_add_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if(EINSUMS_WITH_TESTS_REGRESSIONS)
    einsums_add_pseudo_target(tests.regressions.modules)
    einsums_add_pseudo_dependencies(tests.regressions tests.regressions.modules)
  endif()

  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(tests.performance.modules)
    einsums_add_pseudo_dependencies(tests.performance tests.performance.modules)
  endif()

  if(EINSUMS_WITH_TESTS_HEADERS)
    einsums_add_pseudo_target(tests.headers.modules)
    einsums_add_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()

# Generate a file that lists all enabled modules for checks that might be optional based on the
# presence of a module or not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

# List of sub-directories to work through
set(EINSUMS_LIBS
        Einsums
        CACHE INTERNAL "List of Einsums libraries" FORCE
)

foreach (lib ${EINSUMS_LIBS})
  add_library(${lib} SHARED src/dummy.cpp)
  set_target_properties(
    ${lib}
    PROPERTIES VERSION ${EINSUMS_VERSION}
               SOVERSION ${EINSUMS_SOVERSION}
               CLEAN_DIRECT_OUTPUT 1
               OUTPUT_NAME ${lib}
          FOLDER "Core"
  )
  install(
    TARGETS ${lib}
    EXPORT einsums_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${lib}
  )
  einsums_export_targets(${lib})

  add_subdirectory(${lib})
endforeach()

set(PYTHON_LIBS
        EinsumsPy
        CACHE INTERNAL "List of Python libraries for Einsums" FORCE
)

if(APPLE)
  set(PYTHON_LIB_TYPE MODULE)
else()
  set(PYTHON_LIB_TYPE SHARED)
endif()

if(EINSUMS_BUILD_PYTHON STREQUAL "ON")
  include(Einsums_ExtendWithPython)

  if (NOT EINSUMS_PYTHON_VERSION)
    find_package(
	    Python3 3.10
	    COMPONENTS Interpreter Development NumPy
            REQUIRED
    )
  else ()
    find_package(
            Python3 "${EINSUMS_PYTHON_VERSION}" EXACT
	    COMPONENTS Interpreter Development NumPy
            REQUIRED
    )
  endif ()

  # find_package(pybind11 REQUIRED)
  include(Einsums_SetupPybind11)

  unset(EINSUMS_PYTHON_VERSION)
  foreach (lib ${PYTHON_LIBS})
    set(EINSUMS_PYTHON_LIB_NAME ${lib})
    add_library(${lib} SHARED src/dummy.cpp)
    set_target_properties(
      ${lib}
      PROPERTIES VERSION ${EINSUMS_VERSION}
            SOVERSION ${EINSUMS_SOVERSION}
            CLEAN_DIRECT_OUTPUT 1
            OUTPUT_NAME ${lib}
            FOLDER "Core"
    )

    einsums_extend_with_python(${lib})

    install(
      TARGETS ${lib}
      EXPORT einsums_targets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/einsums
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/einsums
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/einsums COMPONENT ${lib}
    )
    einsums_export_targets(${lib})

    add_subdirectory(${lib})
  endforeach()
elseif(EINSUMS_BUILD_PYTHON MATCHES "3(\.[0-9]+)*")
  foreach(version ${EINSUMS_BUILD_PYTHON})
    set(EINSUMS_PYTHON_VERSION ${version})

    foreach (lib ${PYTHON_LIBS})
      set(EINSUMS_PYTHON_LIB_NAME ${lib}-${version})
      add_library(${lib}-${version} ${PYTHON_LIB_TYPE} src/dummy.cpp)
      set_target_properties(
        ${lib}-${version}
        PROPERTIES VERSION ${EINSUMS_VERSION}
              SOVERSION ${EINSUMS_SOVERSION}
              CLEAN_DIRECT_OUTPUT 1
              OUTPUT_NAME ${lib}
              FOLDER "Core"
      )
      install(
        TARGETS ${lib}-${version}
        EXPORT einsums_targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/einsums
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/einsums
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/einsums COMPONENT ${lib}
      )
      einsums_export_targets(${lib}-${version})

      add_subdirectory(${lib} ${lib}-${version})
    endforeach()
  endforeach()
endif()

# einsums_add_module populates EINSUMS_ENABLED_MODULES
foreach(module ${EINSUMS_ENABLED_MODULES})
  string(TOUPPER ${module} uppercase_module)

  set(MODULE_ENABLED_MODULE_DEFINES
          "${MODULE_ENABLED_MODULE_DEFINES}#define EINSUMS_HAVE_MODULE_${uppercase_module}\n"
  )
endforeach()

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/templates/ModulesEnabled.hpp.in"
        "${PROJECT_BINARY_DIR}/libs/Einsums/Config/include/Einsums/Config/ModulesEnabled.hpp" @ONLY
)

if(EINSUMS_WITH_ITTNOTIFY)
  target_link_libraries(Einsums PUBLIC einsums_internal::amplifier)
endif()
