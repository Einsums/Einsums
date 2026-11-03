#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# GPU abstraction module — always builds.
# Uses mock (CPU) backend when no GPU is available, CUDA or MPS when enabled.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(GPUHeaders
    Einsums/GPU/Platform.hpp
    Einsums/GPU/Error.hpp
    Einsums/GPU/Types.hpp
    Einsums/GPU/Runtime.hpp
    Einsums/GPU/Stream.hpp
    Einsums/GPU/BLAS.hpp
    Einsums/GPU/Solver.hpp
    Einsums/GPU/DeviceVector.hpp
    Einsums/GPU/MPSBackend.hpp
)

set(GPUSources
    Runtime.cpp
    Stream.cpp
    BLAS.cpp
    Solver.cpp
)

# Vendor-specific dependencies and MPS Objective-C++ sources
if(EINSUMS_WITH_CUDA)
  set(GPU_VENDOR_DEPS CUDA::cublas CUDA::cusolver CUDA::cudart)
elseif(EINSUMS_WITH_MPS)
  list(APPEND GPUSources MPSBackend.mm)
  # The Apple frameworks are linked below, privately, rather than as DEPENDENCIES.
  set(GPU_VENDOR_DEPS)
else()
  # Mock backend: no vendor deps, uses CPU BLAS
  set(GPU_VENDOR_DEPS)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums GPU
  PYBIND
  SOURCES ${GPUSources}
  HEADERS ${GPUHeaders}
  DEPENDENCIES ${GPU_VENDOR_DEPS}
  MODULE_DEPENDENCIES Einsums_Config Einsums_CXX23 Einsums_Errors Einsums_Logging Einsums_BLAS Einsums_BLASVendor Einsums_Profile
  CMAKE_SUBDIRS examples tests
)

# Only MPSBackend.mm uses Metal, MetalPerformanceShaders and Foundation, and no public header
# includes them, so they are libEinsums's own link and not part of what a consumer links. Name them
# rather than link the paths find_library found: those paths are in the SDK of the machine that built
# Einsums, and a static libEinsums still exports its private links for consumers to repeat.
if(EINSUMS_WITH_MPS)
  target_link_libraries(
    Einsums PRIVATE "$<LINK_LIBRARY:FRAMEWORK,Metal>"
                    "$<LINK_LIBRARY:FRAMEWORK,MetalPerformanceShaders>"
                    "$<LINK_LIBRARY:FRAMEWORK,Foundation>"
  )
endif()

# MPS: compile the .mm file as Objective-C++ with ARC.
#
# The visibility flags here are load-bearing — without them, MPSBackend.mm
# ends up exporting libc++ weak symbols (e.g. ``std::piecewise_construct``)
# with default visibility while the rest of the project is built with
# hidden visibility (see EINSUMS_WITH_HIDDEN_VISIBILITY in root CMakeLists).
# The linker then resolves direct-access references in other TUs to the
# default-visibility copy from this .mm TU, producing the warning flood:
#   ld: warning: direct access ... to global weak symbol ... means the
#   weak symbol cannot be overridden at runtime.
# Pinning ``-fvisibility=hidden -fvisibility-inlines-hidden`` matches the
# rest of the project and clears the warnings.
if(EINSUMS_WITH_MPS)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MPSBackend.mm
    PROPERTIES
      COMPILE_FLAGS "-fobjc-arc -fvisibility=hidden -fvisibility-inlines-hidden"
  )
endif()
