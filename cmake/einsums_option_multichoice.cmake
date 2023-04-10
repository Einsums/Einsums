#
# Creates a string cache variable with multiple choices
#
# Usage:
#    einsums_option_multichoice(VAR "Description" DEFAULT_VALUE Value1 Value2 ... ValueN)
# Output:
#   VAR is set in the cache and in the caller's scope. The caller can assume
#   that it is always one of the provided values, converted to uppercase.
#
# Main benefit is that the list of allowed values only needs to be provided
# once, and gets used in multiple contexts:
#   1. It is automatically added to the description.
#   2. It is set as the STRINGS property of the created cache variable for use
#      with CMake GUIs.
#   3. The user-provided value is checked against the list, and a fatal error
#      is produced if the value is not known.  The caller does not need to
#      produce good error messages in cases where it may be necessary to check
#      for the validity again.
# As a special case, any "[built-in]" string in the allowed values is ignored
# when checking the user-provided value, but is added to all user-visible
# messages.
#
# It appears that ccmake does not use the STRINGS property, but perhaps some
# day...
#
function(EINSUMS_OPTION_MULTICHOICE NAME DESCRIPTION DEFAULT)
    # Some processing of the input values
    string(REPLACE ";" ", " _allowed_comma_separated "${ARGN}")
    set(_description "${DESCRIPTION}. Pick one of: ${_allowed_comma_separatred}")
    string(REPLACE "[built-in]" "" _allowed "${ARGN}")

    # Set the cache properties
    set(${NAME} ${DEFAULT} CACHE STRING "${_description}")
    set_property(CACHE ${NAME} PROPERTY STRINGS ${_allowed})

    # Check that the value is one of the allowed
    set(_org_value "${${NAME}}")
    string(TOUPPER "${${NAME}}" ${NAME})
    string(TOUPPER "${_allowed}" _allowed_as_upper)
    list(FIND _allowed_as_upper "${${NAME}}" _found_index)
    if (_found_index EQUAL -1)
        message(FATAL_ERROR "Invalid value for ${NAME}: ${_org_value}.  "
                            "Pick one of: ${_allowed_comma_separatred}")
    endif()

    # Always provide the upper-case value to the caller
    set(${NAME} "${${NAME}}" PARENT_SCOPE)
endfunction()

#
# Hides or shows a cache value based on conditions
#
# Usage:
#   einsums_add_cache_dependency(VAR TYPE CONDITIONS VALUE)
# where
#   VAR            is a name of a cached variable
#   TYPE           is the type of VAR
#   CONDITIONS     is a list of conditional expressions (see below)
#   VALUE          is a value that is set to VAR if CONDITIONS is not satisfied
#
# Evaluates each condition in CONDITIONS, and if any of them is false,
# VAR is marked internal (hiding it from the user) and its value is set to
# VALUE. The previous user-set value of VAR is still remembered in the cache,
# and used when CONDITIONS become true again.
#
# The conditions is a semicolon-separated list of conditions as specified for
# CMake if() statements, such as "EINSUMS_FFT_LIBRARY STREQUAL FFTW3",
# "NOT EINSUMS_MPI" or "EINSUMS_MPI;NOT EINSUMS_DOUBLE".  Note that quotes within the
# expressions don't work for some reason (even if escaped).
#
# The logic is adapted from cmake_dependent_option().
#
function(EINSUMS_ADD_CACHE_DEPENDENCY NAME TYPE CONDITIONS FORCED_VALUE)
    set(_available TRUE)
    foreach (_cond ${CONDITIONS})
        string(REGEX REPLACE " +" ";" _cond_as_list ${_cond})
        if (${_cond_as_list})
        else()
            set(_available FALSE)
        endif()
    endforeach()
    if (_available)
        set_property(CACHE ${NAME} PROPERTY TYPE ${TYPE})
    else()
        set(${NAME} "${FORCED_VALUE}" PARENT_SCOPE)
        set_property(CACHE ${NAME} PROPERTY TYPE INTERNAL)
    endif()
endfunction()

# Works like cmake_dependent_option(), but allows for an arbitrary cache value
# instead of only an ON/OFF option
#
# Usage:
#   einsums_dependent_cache_variable(VAR "Description" TYPE DEFAULT CONDITIONS)
#
# Creates a cache variable VAR with the given description, type and default
# value.  If any of the conditions listed in CONDITIONS is not true, then
# the cache variable is marked internal (hiding it from the user) and the
# value of VAR is set to DEFAULT.  The previous user-set value of VAR is still
# remembered in the cache, and used when CONDITIONS become true again.
# Any further changes to the variable can be done with simple set()
# (or set_property(CACHE VAR PROPERTY VALUE ...) if the cache needs to be
# modified).
#
# See einsums_add_cache_dependency() on how to specify the conditions.
#
macro(EINSUMS_DEPENDENT_CACHE_VARIABLE NAME DESCRIPTION TYPE DEFAULT CONDITIONS)
    set(${NAME} "${DEFAULT}" CACHE ${TYPE} "${DESCRIPTION}")
    einsums_add_cache_dependency(${NAME} ${TYPE} "${CONDITIONS}" "${DEFAULT}")
endmacro()

# Works like cmake_dependent_option(), but reuses the code from
# einsums_dependent_cache_variable() to make sure both behave the same way.
macro(EINSUMS_DEPENDENT_OPTION NAME DESCRIPTION DEFAULT CONDITIONS)
    einsums_dependent_cache_variable(${NAME} "${DESCRIPTION}" BOOL "${DEFAULT}" "${CONDITIONS}")
endmacro()

# Sets a boolean variable based on conditions
#
# Usage:
#   einsums_set_boolean(VAR CONDITIONS)
#
# Sets VAR to ON if all conditions listed in CONDITIONS are true, otherwise
# VAR is set OFF.
#
# See einsums_add_cache_dependency() on how to specify the conditions.
#
function (EINSUMS_SET_BOOLEAN NAME CONDITIONS)
    set(${NAME} ON)
    foreach(_cond ${CONDITIONS})
        string(REGEX REPLACE " +" ";" _cond_as_list ${_cond})
        if (${_cond_as_list})
        else()
            set(${NAME} OFF)
        endif()
    endforeach()
    set(${NAME} ${${NAME}} PARENT_SCOPE)
endfunction()

# Checks if one or more variables have changed since last call to this function
#
# Usage:
#   einsums_check_if_changed(RESULT VAR1 VAR2 ... VARN)
#
# Sets RESULT to true if any of the given variables VAR1 ... VARN has
# changed since the last call to this function for that variable.
# Changes are tracked also across CMake runs.
function(EINSUMS_CHECK_IF_CHANGED RESULT)
    set(_result FALSE)
    foreach (_var ${ARGN})
        if (NOT "${${_var}}" STREQUAL "${${_var}_PREVIOUS_VALUE}")
            set(_result TRUE)
        endif()
        set(${_var}_PREVIOUS_VALUE "${${_var}}" CACHE INTERNAL
            "Previous value of ${_var} for change tracking")
    endforeach()
    set(${RESULT} ${_result} PARENT_SCOPE)
endfunction()
