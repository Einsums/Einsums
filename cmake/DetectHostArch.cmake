# Taken from https://github.com/scivision/cmake-cpu-detect/blob/main/DetectHostArch.cmake

#=============================================================================
# Copyright 2010-2016 Matthias Kretz <kretz@kde.org>
# Copyright 2015 ArangoDB GmbH
# Copyright 2020 Michael Hirsch <www.scivision.dev>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the names of contributing organizations nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================
#
# DetectHostArch()
#
# For current Intel CPUs, attempt to detect host architecture on Linux, MacOS and Windows.
# Useful for Intel compilers "-march="
# https://software.intel.com/content/www/us/en/develop/documentation/fortran-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/compiler-option-details/code-generation-options/arch.html
# and as a general example of extracting CPU arch to set CPU arch specific flags on other compilers.
#
# To keep things manageable, we omitted Intel CPUs older than Sandy Bridge ca. 2010
#
# Result Variables
# ----------------
#
# HOST_ARCH
#
#==============================================================================

function(GetHostCPUInfo)

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
  set(_file /proc/cpuinfo)
  if(NOT EXISTS ${_file})
    message(STATUS "Could not find ${_file}")
    return()
  endif()

  file(READ ${_file} _info)
  string(REGEX REPLACE ".*vendor_id[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" CPU_VENDOR_ID "${_info}")
  string(REGEX REPLACE ".*cpu family[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" CPU_FAMILY "${_info}")
  string(REGEX REPLACE ".*model[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" CPU_MODEL "${_info}")
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  find_program(_sys NAMES sysctl PATHS /usr/sbin)
  if(NOT _sys)
    message(STATUS "Could not find ${_sys}")
    return()
  endif()

  execute_process(COMMAND ${_sys} -n machdep.cpu.vendor OUTPUT_VARIABLE CPU_VENDOR_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${_sys} -n machdep.cpu.model  OUTPUT_VARIABLE CPU_MODEL OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${_sys} -n machdep.cpu.family OUTPUT_VARIABLE CPU_FAMILY OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
  get_filename_component(CPU_VENDOR_ID "[HKEY_LOCAL_MACHINE\\Hardware\\Description\\System\\CentralProcessor\\0;VendorIdentifier]" NAME CACHE)
  get_filename_component(_cpu_id "[HKEY_LOCAL_MACHINE\\Hardware\\Description\\System\\CentralProcessor\\0;Identifier]" NAME CACHE)
  message(TRACE "CPU ID: ${_cpu_id}")

  string(REGEX REPLACE ".* Family ([0-9]+) .*" "\\1" CPU_FAMILY "${_cpu_id}")
  string(REGEX REPLACE ".* Model ([0-9]+) .*" "\\1" CPU_MODEL "${_cpu_id}")
else()
  message(STATUS "Unknown operating system ${CMAKE_SYSTEM_NAME}")
  return()
endif()

set(CPU_VENDOR_ID ${CPU_VENDOR_ID} PARENT_SCOPE)
set(CPU_FAMILY ${CPU_FAMILY} PARENT_SCOPE)
set(CPU_MODEL ${CPU_MODEL} PARENT_SCOPE)

endfunction(GetHostCPUInfo)


function(_decode_intel)
# https://en.wikichip.org/wiki/intel/cpuid
# https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/compiler-option-details/code-generation-options/march.html

set(HOST_ARCH)

if(CPU_FAMILY EQUAL 6)
  if(CPU_MODEL EQUAL 133)
    set(HOST_ARCH "knm")
  elseif(CPU_MODEL EQUAL 87)
    set(HOST_ARCH "knl")
  elseif(CPU_MODEL EQUAL 134)
    set(HOST_ARCH "tremont")
  elseif(CPU_MODEL EQUAL 122)
    set(HOST_ARCH "goldmont-plus")
  elseif(CPU_MODEL EQUAL 92)
    set(HOST_ARCH "goldmont")
  elseif(CPU_MODEL EQUAL 90 OR CPU_MODEL EQUAL 76)
    set(HOST_ARCH "silvermont")
  elseif(CPU_MODEL EQUAL 140)
    set(HOST_ARCH "tigerlake")
  elseif(CPU_MODEL EQUAL 106 OR CPU_MODEL EQUAL 108)
    set(HOST_ARCH "icelake-server")
  elseif(CPU_MODEL EQUAL 126 OR CPU_MODEL EQUAL 125)
    set(HOST_ARCH "icelake-client")
  elseif(CPU_MODEL EQUAL 102)
    set(HOST_ARCH "cannonlake")
  elseif(CPU_MODEL EQUAL 142 OR CPU_MODEL EQUAL 158)
    set(HOST_ARCH "coffeelake")
  elseif(CPU_MODEL EQUAL 85)
    set(HOST_ARCH "skylake-avx512")
  elseif(CPU_MODEL EQUAL 78 OR CPU_MODEL EQUAL 94)
    set(HOST_ARCH "skylake")
  elseif(CPU_MODEL EQUAL 61 OR CPU_MODEL EQUAL 71 OR CPU_MODEL EQUAL 79 OR CPU_MODEL EQUAL 86)
    set(HOST_ARCH "broadwell")
  elseif(CPU_MODEL EQUAL 60 OR CPU_MODEL EQUAL 69 OR CPU_MODEL EQUAL 70 OR CPU_MODEL EQUAL 63)
    set(HOST_ARCH "haswell")
  elseif(CPU_MODEL EQUAL 58 OR CPU_MODEL EQUAL 62)
    set(HOST_ARCH "ivybridge")
  elseif(CPU_MODEL EQUAL 42 OR CPU_MODEL EQUAL 45)
    set(HOST_ARCH "sandybridge")
  endif()
endif()

if(HOST_ARCH)
  set(HOST_ARCH ${HOST_ARCH} CACHE STRING "CPU Architecture")
endif()

endfunction(_decode_intel)


function(detect_host_arch)

GetHostCPUInfo()

if(CPU_VENDOR_ID STREQUAL "GenuineIntel")
  _decode_intel()
  message(VERBOSE "CPU: ${HOST_ARCH} ${CPU_VENDOR_ID} ${CPU_FAMILY} ${CPU_MODEL}")
endif()



# --- capability check
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(HOST_FLAGS -march=native -mtune=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(HOST_FLAGS -march=native -mtune=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  set(HOST_FLAGS -march=native -mtune=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  if(WIN32)
    set(HOST_FLAGS /QxHost /tune:${HOST_ARCH})
  else(WIN32)
    set(HOST_FLAGS -xHost -mtune=${HOST_ARCH})
  endif()
endif()

set(CMAKE_REQUIRED_FLAGS ${HOST_FLAGS})
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)

set(_code "#include <immintrin.h>
__m256i i;
int main(void) {__m256i a = _mm256_abs_epi16(i); return 0;}")
check_cxx_source_compiles("${_code}" HAS_AVX2)

set(_code "#include <immintrin.h>
int main(void) {__m256 a = _mm256_setzero_ps(); return 0;}")
check_cxx_source_compiles("${_code}" HAS_AVX)

if(CMAKE_C_COMPILER_ID STREQUAL GNU)
  set(CMAKE_REQUIRED_FLAGS ${HOST_FLAGS} -ftree-vectorize -mfpu=neon)
endif()
set(_code "#include \"arm_neon.h\"
int main(void){float32x4_t v1 = { 1.0, 2.0, 3.0, 4.0 }; return 0;}")
check_c_source_compiles("${_code}" HAS_NEON)
if(HAS_NEON)
  if(CMAKE_C_COMPILER_ID STREQUAL GNU)
    list(APPEND HOST_FLAGS -mfpu=neon)
  endif()
endif()

set(HOST_ARCH ${HOST_ARCH} PARENT_SCOPE)
set(HOST_FLAGS ${HOST_FLAGS} PARENT_SCOPE)
set(HAS_AVX2 ${HAS_AVX2} CACHE BOOL "CPU has AVX2 instructions")
set(HAS_AVX ${HAS_AVX} CACHE BOOL "CPU has AVX instructions")
set(HAS_NEON ${HAS_NEON} CACHE BOOL "CPU has Neon instructions")

endfunction(detect_host_arch)