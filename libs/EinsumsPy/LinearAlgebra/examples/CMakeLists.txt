#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra)
  einsums_add_pseudo_dependencies(
          Examples.Modules Examples.Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra
  )
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra
    )
  endif()
else()
  return()
endif()

set(example_programs)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  einsums_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Modules/${EINSUMS_PYTHON_LIB_NAME}/LinearAlgebra"
    NOINSTALL
  )

  einsums_add_example_target_dependencies(
          "Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra" ${example_program}
  )

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
            "Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra" ${example_program}
            ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
