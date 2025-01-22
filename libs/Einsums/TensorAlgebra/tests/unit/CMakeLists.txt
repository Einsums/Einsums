#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TensorAlgebraTests
    Dot
    Einsum1
    Einsum2
    Einsum3
    Einsum4
    Einsum5
    Einsum6
    Einsum7
    Element
    Hadamard
    Indices
    KhatriRao
    MixedPrecision
    OuterProduct
    Sort
    Transpose
    UniquePointer
)

foreach(test ${TensorAlgebraTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    FOLDER ${folder_name}
          NOINSTALL
  )

  einsums_add_unit_test("Modules.TensorAlgebra" ${test})
endforeach()

if(EINSUMS_WITH_GPU_SUPPORT)
  set(TensorAlgebraTests2 Dot)

  foreach(test ${TensorAlgebraTests2})
    set(sources ${test}.hip)

    if(EINSUMS_WITH_CUDA)
      set_source_files_properties(${test}.hip PROPERTIES LANGUAGE CUDA)
    endif()

    source_group("Source Files" FILES ${sources})

    einsums_add_executable(
      ${test}_gpu_test INTERNAL_FLAGS
      SOURCES ${sources} ${${test}_FLAGS}
      FOLDER ${folder_name}
    )

    einsums_add_unit_test("Modules.TensorAlgebra" ${test}_gpu)
  endforeach()
endif()
