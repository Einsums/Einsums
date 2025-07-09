..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

Compiler selection and customizing a build
******************************************

Selecting a specific compiler
=============================

CMake supports the standard environment variable ``CXX`` to select a specific C++ compiler.
This environment variable is documented in the `CMake docs
<https://cmake.org/cmake/help/latest/envvar/CC.html>`__.

Note that environment variables only get applied from a clean build, because they affect
the configuration stage. An incremental rebuild does not react to changes in environment
variables.

CMake also supports passing the ``-DCMAKE_CXX_COMPILER=`` `variable
<https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html>`__
to the command-line call to ``cmake``.

Selecting build type
====================

CMake natively supports four build types: ``Debug``, ``Release``, ``MinSizeRel``, and ``RelWithDebInfo``.

1. ``Release``: high optimization level, no debug info, code or asserts.

2. ``Debug``: No optimization, asserts enabled, [custom debug (output) code enabled],
   debug info included in executable (so you can step through the code with a
   debugger and have address to source-file:line-number translation).

3. ``RelWithDebInfo``: optimized, *with* debug info, but no debug (output) code or asserts.

4. ``MinSizeRel``: same as Release but optimizing for size rather than speed.

Einsums provides additional build types: ``ASAN``, ``MSAN``, and ``UBSAN``. These are not used very often and currently are not
guaranteed to work.

5. ``ASAN``: Address Sanitizer (aka ASan) is a memory error detector for C/C++. It finds:

    * Use after free (dangling pointer dereference)

    * Heap buffer overflow

    * Stack buffer overflow

    * Global buffer overflow

    * Use after return

    * Use after scope

    * Initialization order bugs

    * Memory leaks

6. ``MSAN``: Memory Sanitizer (aka MSan) is a detector of uninitialized memory reads in C/C++ programs.
   Uninitialized values occur when stack- or heap-allocated memory is read before it is written.
   MSan detects cases where such values affect program execution.
   MSan is bit-exact: it can track uninitialized bits in a bitfield.
   It will tolerate copying of uninitialized memory, and also simple logic and arithmetic operations with it.

   In general, MSan silently tracks the spread of uninitialized data in memory, and reports a warning when a
   code branch is taken (or not taken) depending on an uninitialized value.

   MSan implements a subset of functionality found in Valgrind (Memcheck tool). It is significantly faster
   than Memcheck.

7. ``UBSAN``: Undefined Behavior Sanitizer (UBSan) is a fast undefined behavior detector. UBSan modifies
   the program at compile-time to catch various kinds of undefined behavior during program execution, for
   example:

   * Array subscript out of bounds, where the bounds can be statically determined

   * Bitwise shifts that are out of bounds for their data type

   * Dereferencing misaligned or null pointers

   * Signed integer overflow

   * Conversion to, from, or between floating-point types which would overflow the destination

See :ref:`the page on CMake variables <cmake_variables>` for a list of options that can be passed to
CMake when configuring.

Building for GPU
================

When building for GPU, special steps need to be taken. First, set the ``-DEINSUMS_WITH_HIP=ON`` or ``-DEINSUMS_WITH_CUDA=ON`` flag
to enable GPU compilation. Then, a HIP-compatible compiler must be specified. AMD's fork of Clang that comes with HIP is a good
choice. Then, in order for CMake to find the appropriate libraries, ``-DCMAKE_HIP_COMPILER_ROCM_ROOT`` needs to be set to
the root directory of the ROCm installation. On Linux, this is often ``/opt/rocm``. Since this flag has a long name, 
Einsums provides an alias: ``-DHIP_ROCM_ROOT``. You may set either one and the other will be populated. Then, simply configure
and build. If you get configuration errors about being unable to find certain HIP/ROCm libraries, simply set the variables that
CMake is asking for. The CMake files can be found under ``${HIP_ROCM_ROOT}/lib/cmake``.