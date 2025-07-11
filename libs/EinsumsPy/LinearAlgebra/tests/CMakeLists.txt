#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_Message)

if(EINSUMS_WITH_TESTS)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(Tests.Unit.Modules.EinsumsPy.LinearAlgebra)
    einsums_add_pseudo_dependencies(Tests.Unit.Modules Tests.Unit.Modules.EinsumsPy.LinearAlgebra)
    add_subdirectory(unit)
  endif()

  if(EINSUMS_WITH_TESTS_REGRESSIONS)
    einsums_add_pseudo_target(Tests.Regressions.Modules.EinsumsPy.LinearAlgebra)
    einsums_add_pseudo_dependencies(
      Tests.Regressions.Modules Tests.Regressions.Modules.EinsumsPy.LinearAlgebra
    )
    add_subdirectory(regressions)
  endif()

  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(Tests.Performance.Modules.EinsumsPy.LinearAlgebra)
    einsums_add_pseudo_dependencies(
      Tests.Performance.Modules Tests.Performance.Modules.EinsumsPy.LinearAlgebra
    )
    add_subdirectory(performance)
  endif()

  if(EINSUMS_WITH_TESTS_HEADERS)
    einsums_add_header_tests(
      Modules.${EINSUMS_PYTHON_LIB_NAME}.LinearAlgebra
      HEADERS ${LinearAlgebraHeaders}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      DEPENDENCIES ${EINSUMS_PYTHON_LIB_NAME}_LinearAlgebra pybind11::embed
    )
  endif()
endif()
