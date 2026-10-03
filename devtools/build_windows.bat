@echo off
REM Copyright (c) The Einsums Developers. All rights reserved.
REM Licensed under the MIT License. See LICENSE.txt in the project root for license information.
REM
REM Configure Einsums on Windows (clang-cl from conda + Ninja).
REM Prerequisite: conda activate (einsums-dev or equivalent) in cmd.

REM Clear conda feedstock flags so CMAKE_CXX_STANDARD / CMAKE_BUILD_TYPE win.
set "CXXFLAGS="
set "CFLAGS="
set "CPPFLAGS="

REM vs2022_compiler_vars sets these for the VS generator. Ninja rejects any
REM platform specification (including an empty value), so they must be unset —
REM not set to empty via the CMake preset environment map.
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
set "CMAKE_GENERATOR="

cmake --preset windows-clang-cl
echo.
echo Exit code: %ERRORLEVEL%
exit /b %ERRORLEVEL%
