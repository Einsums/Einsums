#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(examples.modules.${EINSUMS_PYTHON_LIB_NAME}.TestUtils)
  einsums_add_pseudo_dependencies(examples.modules examples.modules.${EINSUMS_PYTHON_LIB_NAME}.TestUtils)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(tests.examples.modules.${EINSUMS_PYTHON_LIB_NAME}.TestUtils)
    einsums_add_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.${EINSUMS_PYTHON_LIB_NAME}.TestUtils
    )
  endif()
endif()