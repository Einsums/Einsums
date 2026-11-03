#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# FindTargetLAPACK.cmake
# ----------------------
#
# LAPACK cmake module to wrap FindLAPACK.cmake in a target.
#
# This module sets the following variables in your project: ::
#
# TargetLAPACK_FOUND - true if BLAS/LAPACK found on the system TargetLAPACK_MESSAGE - status message
# with BLAS/LAPACK library path list
#
# This module *unsets* the following conventional LAPACK variables so as to force using the target:
# ::
#
# LAPACK_FOUND LAPACK_LIBRARIES
#
# In order of decreasing precedence, this module returns in a target ``tgt::lapack`` (1) the
# libraries passed through CMake variable LAPACK_LIBRARIES, (2) the libraries defined in a
# detectable TargetLAPACKConfig.cmake file (skip via CMAKE_DISABLE_FIND_PACKAGE_TargetLAPACK), or
# (3) the libraries detected by the usual FindLAPACK.cmake module.
#
# For (3), MKL is tried first, then OpenBLAS (lp64, then ilp64), a generic BLAS, and last whatever
# FindBLAS picks on its own. These inputs narrow that search: ::
#
# BLA_VENDOR - set by the caller, it replaces the passes with one search for that vendor
# TargetLAPACK_INT_INTERFACE - lp64 or ilp64 keeps only the passes of that integer width
# TargetLAPACK_SKIP_MKL - true skips the MKL attempt
#
# Einsums specialization * sets property VENDOR for MKL-ness * sets property INT_INTERFACE for
# lp64/ilp64 * uses targets, not variables, from FindLAPACK (CMake 3.18)

set(PN TargetLAPACK)

# 1st precedence - libraries passed in through -DLAPACK_LIBRARIES
if(LAPACK_LIBRARIES)
  if(NOT ${PN}_FIND_QUIETLY)
    message(STATUS "LAPACK detection suppressed.")
  endif()

  set(_VENDOR "All")
  foreach(_l IN LISTS LAPACK_LIBRARIES)
    get_filename_component(_lname ${_l} NAME)
    if(${_lname} MATCHES "mkl")
      set(_VENDOR "MKL")
      break()
    elseif(${_lname} MATCHES "openblas")
      set(_VENDOR "OpenBLAS")
      break()
    endif()
  endforeach()

  add_library(tgt::lapack INTERFACE IMPORTED)
  set_property(TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
  set_property(TARGET tgt::lapack PROPERTY VENDOR ${_VENDOR})
  set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE lp64) # TODO assumption!
else()
  # 2nd precedence - target already prepared and findable in TargetLAPACKConfig.cmake
  if(NOT "${CMAKE_DISABLE_FIND_PACKAGE_${PN}}")
    find_package(TargetLAPACK QUIET CONFIG)
  endif()
  if(TARGET tgt::lapack)
    if(NOT ${PN}_FIND_QUIETLY)
      message(STATUS "TargetLAPACKConfig detected.")
    endif()
  else()
    # 3rd precedence - usual variables from FindLAPACK.cmake
    # The caller's BLA_VENDOR and BLA_SIZEOF_INTEGER are put back once the passes below have used
    # these names.
    unset(_tl_caller_vendor)
    unset(_tl_caller_sizeof)
    if(DEFINED BLA_VENDOR)
      set(_tl_caller_vendor "${BLA_VENDOR}")
    endif()
    if(DEFINED BLA_SIZEOF_INTEGER)
      set(_tl_caller_sizeof "${BLA_SIZEOF_INTEGER}")
    endif()

    # A vendor the caller names is searched for as asked, MKL's FindBLAS names included.
    set(_tl_try_mkl TRUE)
    if(TargetLAPACK_SKIP_MKL OR DEFINED _tl_caller_vendor)
      set(_tl_try_mkl FALSE)
    endif()
    if(_tl_try_mkl AND NOT MSVC)
      find_package(MKL CONFIG)
    endif()
    if(_tl_try_mkl AND TARGET MKL::MKL)
      add_library(tgt::lapack INTERFACE IMPORTED)
      set_property(TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES MKL::MKL)
      set_property(TARGET tgt::lapack PROPERTY VENDOR "MKL")
      set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE ${MKL_INTERFACE})

    else()
      # Each pass is vendor:BLA_SIZEOF_INTEGER:INT_INTERFACE. The generic and FindBLAS-chosen
      # passes cannot tell the integer width, so they assume lp64.
      if(DEFINED _tl_caller_vendor)
        if(DEFINED _tl_caller_sizeof)
          set(_tl_sizeof "${_tl_caller_sizeof}")
        elseif(TargetLAPACK_INT_INTERFACE STREQUAL "ilp64")
          set(_tl_sizeof 8)
        else()
          set(_tl_sizeof ANY)
        endif()
        if(_tl_sizeof STREQUAL "8")
          set(_tl_passes "${_tl_caller_vendor}:8:ilp64")
        else()
          set(_tl_passes "${_tl_caller_vendor}:${_tl_sizeof}:lp64")
        endif()
      else()
        set(_tl_passes OpenBLAS:4:lp64 OpenBLAS:8:ilp64 Generic:ANY:lp64 All:ANY:lp64)
        if(TargetLAPACK_INT_INTERFACE)
          if(NOT TargetLAPACK_INT_INTERFACE MATCHES "^i?lp64$")
            message(
              FATAL_ERROR
                "TargetLAPACK_INT_INTERFACE is '${TargetLAPACK_INT_INTERFACE}'; it must be lp64 or ilp64"
            )
          endif()
          list(FILTER _tl_passes INCLUDE REGEX ":${TargetLAPACK_INT_INTERFACE}$")
        endif()
      endif()

      # Only the last pass is REQUIRED and reports its misses; the earlier ones are expected to
      # miss whenever a later one is what the environment has.
      list(LENGTH _tl_passes _tl_npasses)
      set(_tl_ipass 0)
      foreach(_tl_pass IN LISTS _tl_passes)
        math(EXPR _tl_ipass "${_tl_ipass} + 1")
        string(REPLACE ":" ";" _tl_fields "${_tl_pass}")
        list(GET _tl_fields 0 BLA_VENDOR)
        list(GET _tl_fields 1 BLA_SIZEOF_INTEGER)
        list(GET _tl_fields 2 _tl_int_interface)
        if(_tl_ipass EQUAL _tl_npasses)
          set(_tl_mode REQUIRED)
        else()
          set(_tl_mode QUIET)
        endif()
        find_package(BLAS ${_tl_mode} MODULE)
        find_package(LAPACK ${_tl_mode} MODULE)

        if((TARGET BLAS::BLAS) AND (TARGET LAPACK::LAPACK))
          add_library(tgt::lapack INTERFACE IMPORTED)
          set_property(
            TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES LAPACK::LAPACK BLAS::BLAS
          )
          set_property(TARGET tgt::lapack PROPERTY VENDOR ${BLA_VENDOR})
          set_property(TARGET tgt::lapack PROPERTY INT_INTERFACE ${_tl_int_interface})
          break()
        endif()
      endforeach()

    endif()

    # The VENDOR set above is BLA_VENDOR, which records what the search asked
    # for rather than what it found. They diverge on Windows: find_package(MKL
    # CONFIG) is skipped under MSVC, the OpenBLAS and Generic passes then fail,
    # and the "All" pass succeeds against MKL while reporting "All". Consumers
    # keyed on the vendor silently take the wrong branch - EINSUMS_HAVE_MKL is
    # never defined, so MKL's xerbla is not routed into our logger, and the FFT
    # default falls back to fftw3 on a build that links MKL. Re-derive it from
    # the libraries actually resolved, matching how the LAPACK_LIBRARIES branch
    # at the top of this file names the vendor.
    if(TARGET tgt::lapack)
      foreach(_l IN LISTS LAPACK_LIBRARIES BLAS_LIBRARIES)
        get_filename_component(_lname "${_l}" NAME)
        if(_lname MATCHES "[mM][kK][lL]")
          set_property(TARGET tgt::lapack PROPERTY VENDOR "MKL")
          break()
        elseif(_lname MATCHES "[oO][pP][eE][nN][bB][lL][aA][sS]")
          set_property(TARGET tgt::lapack PROPERTY VENDOR "OpenBLAS")
          break()
        endif()
      endforeach()
    endif()

    if(NOT ${PN}_FIND_QUIETLY)
      message(STATUS "LAPACK detected.")
    endif()

    unset(BLA_VENDOR)
    unset(BLA_SIZEOF_INTEGER)
    if(DEFINED _tl_caller_vendor)
      set(BLA_VENDOR "${_tl_caller_vendor}")
    endif()
    if(DEFINED _tl_caller_sizeof)
      set(BLA_SIZEOF_INTEGER "${_tl_caller_sizeof}")
    endif()
    unset(LAPACK_FOUND)
    unset(LAPACK_LIBRARIES)
  endif()
endif()

get_property(
  _ill
  TARGET tgt::lapack
  PROPERTY INTERFACE_LINK_LIBRARIES
)
get_property(
  _ven
  TARGET tgt::lapack
  PROPERTY VENDOR
)
get_property(
  _int
  TARGET tgt::lapack
  PROPERTY INT_INTERFACE
)
set(${PN}_MESSAGE "Found LAPACK ${_ven}w/${_int}: ${_ill}")
if((TARGET tgt::blas) AND (TARGET tgt::lapk))
  get_property(
    _illb
    TARGET tgt::blas
    PROPERTY INTERFACE_LINK_LIBRARIES
  )
  get_property(
    _illl
    TARGET tgt::lapk
    PROPERTY INTERFACE_LINK_LIBRARIES
  )
  set(${PN}_MESSAGE "Found LAPACK ${_ven}w/${_int}: ${_illl};${_illb}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PN} DEFAULT_MSG ${PN}_MESSAGE)
