include(CMakeDependentOption)
include(CMakeParseArguments)

macro(einsums_option NAME TYPE DESCRIPTION DEFAULT)

    set(options ADVANCED)
    set(one_value_args CATEGORY DEPENDS)
    set(multi_value_args STRINGS)
    cmake_parse_arguments(EINSUMS_OPTION "${options}" "${one_value_args}" "${multi_value_args}")

    if ("${TYPE}" STREQUAL "BOOL")
        # Use regular CMake option handling for booleans
        if (NOT EINSUMS_OPTION_DEPENDS)
            option("${NAME}" "${DESCRIPTION}" "${DEFAULT}")
        else()
            cmake_dependent_option("${NAME}" "${DESCRIPTION}" "${DEFAULT}" "${EINSUMS_OPTION_DEPENDS}" OFF)
        endif()
    else()
        if (EINSUMS_OPTION_DEPENDS)
            message(FATAL_ERROR "einsums_option DEPENDS keyword can only be used with BOOL options")
        endif()

        if (NOT DEFINED ${NAME})
            set(${NAME} ${DEFAULT} CACHE ${TYPE} "${DESCRIPTION}" FORCE)
        else()
            get_property(
                _option_is_cache_property 
                CACHE "${NAME}" 
                PROPERTY TYPE 
                SET
            )
            if (NOT _option_is_cache_property)
                set(${NAME} ${DEFAULT} CACHE ${TYPE} "${DESCRIPTION}" FORCE)
                if (EINSUMS_OPTION_ADVANCED)
                    mark_as_advanced(${NAME})
                endif()
            else()
                set_property(CACHE "${NAME}" PROPERTY HELPSTRING "${DESCRIPTION}")
                set_property(CACHE "${NAME}" PROPERTY TYPE "${TYPE}")
            endif()
        endif()

        if (EINSUMS_OPTION_STRINGS)
            if ("${TYPE}" STREQUAL "STRING")
                set_property(CACHE "${NAME}" PROPERTY STRINGS "${EINSUMS_OPTION_STRINGS}")
            else()
                message(FATAL_ERROR "einsums_option STRINGS can only be used if type is STRING")
            endif()
        endif()
    endif()

    if (EINSUMS_OPTION_ADVANCED)
        mark_as_advanced(${NAME})
    endif()

    set_property(GLOBAL APPEND PROPERTY EINSUMS_MODULE_CONFIG ${NAME})
endmacro()