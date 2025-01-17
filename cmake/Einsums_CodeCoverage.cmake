#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Setup coverage for specific target
function(einsums_append_coverage_compiler_flags_to_target name)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping>)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fprofile-instr-generate -fcoverage-mapping>)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:--coverage>)
    target_link_options(${name} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping>)
    target_link_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fprofile-instr-generate -fcoverage-mapping>)
    target_link_options(${name} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-lgcov --coverage>)
endfunction()
