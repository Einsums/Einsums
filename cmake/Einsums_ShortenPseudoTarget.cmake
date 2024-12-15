#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_shorten_pseudo_target target shortened_target)
  einsums_debug("einsums_shorten_pseudo_target" "shortening pseudo target: ${target}")
  if(WIN32)
    set(args)
    foreach(arg ${target})
      string(REPLACE "." ";" elements ${arg})
      list(GET elements -1 arg)
      set(args ${args} ${arg})
    endforeach()
    set(${shortened_target}
        ${args}
            PARENT_SCOPE
    )
    einsums_debug("einsums_shorten_pseudo_target" "shortened pseudo target: ${${shortened_target}}")
  else()
    set(${shortened_target}
        ${target}
            PARENT_SCOPE
    )
  endif()
endfunction()
