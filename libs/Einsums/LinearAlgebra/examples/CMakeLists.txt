#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.Einsums.LinearAlgebra)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.LinearAlgebra)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.LinearAlgebra)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.Einsums.LinearAlgebra
    )
  endif()
else()
  return()
endif()

set(example_programs Dot)

# Highway is agonizingly slow in debug builds
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND example_programs hwy_play hwy_triplet)
endif()

set(hwy_play_FLAGS DEPENDENCIES hwy)
set(hwy_triplet_FLAGS DEPENDENCIES hwy)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  einsums_add_executable(
    LA_${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Modules/LinearAlgebra"
    NOINSTALL
  )
  target_include_directories(LA_${example_program} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  einsums_add_example_target_dependencies("Modules.Einsums.LinearAlgebra" LA_${example_program})

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
      "Modules.Einsums.LinearAlgebra" LA_${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
