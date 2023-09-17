if (EINSUMS_API_DEFINED)
    return()
endif(EINSUMS_API_DEFINED)
set(EINSUMS_API_DEFINED TRUE)

include(${CMAKE_CURRENT_LIST_DIR}/EinsumsAPIInternal.cmake)

set(EINSUMS_LIBRARY_BASE_PATH "${_EINSUMS_LIBRARY_BASE_PATH}")  # The Einsums library base path (relative to CMAKE_INSTALL_PREFIX).
set(EINSUMS_LIBRARY_PATH "${_EINSUMS_LIBRARY_PATH}")            # The Einsums library path (relative to CMAKE_INSTALL_PREFIX).
set(EINSUMS_LIBRARY_ARCHIVE_PATH "${_EINSUMS_LIBRARY_ARCHIVE_PATH}") # The Einsums library archive path (relative to CMAKE_INSTALL_PREFIX).
set(EINSUMS_LIBEXEC_PATH "${_EINSUMS_LIBEXEC_PATH}")            # The Einsums libexec path (relative to CMAKE_INSTALL_PREFIX).
set(EINSUMS_DATA_PATH "${_EINSUMS_DATA_PATH}")                  # The Einsums data path (relative to CMAKE_INSTALL_PREFIX).
set(EINSUMS_DOC_PATH "${_EINSUMS_DOC_PATH}")                    # The Einsums documentation path (relative to CMAKE_INSTALL_PREFIX).
set(EINSUMS_BIN_PATH "${_EINSUMS_BIN_PATH}")                    # The Einsums bin path (relative to CMAKE_INSTALL_PREFIX).

set(EINSUMS_HEADER_INSTALL_PATH "${_EINSUMS_HEADER_INSTALL_PATH}")
set(EINSUMS_CMAKE_INSTALL_PATH "${_EINSUMS_CMAKE_INSTALL_PATH}")


option(EINSUMS_STATIC_BUILD "Build libraries as static libraries" OFF)

function(einsums_output_binary_dir varName)
    if (EINSUMS_MERGE_BINARY_DIR)
        set(${varName} ${EINSUMS_BINARY_DIR} PARENT_SCOPE)
    else()
        set(${varName} ${PROJECT_BINARY_DIR} PARENT_SCOPE)
    endif()
endfunction()

function(add_einsums_library name)
    cmake_parse_arguments(_arg "STATIC;OBJECT;SHARED;FEATURE_INFO;SKIP_PCH"
        "DESTINATION;COMPONENT;SOURCES_PREFIX;BUILD_DEFAULT"
        "CONDITION;DEPENDS;PUBLIC_DEPENDS;DEFINES;PUBLIC_DEFINES;INCLUDES;PUBLIC_INCLUDES;SOURCES;PROPERTIES;PUBLIC_OPTIONS;OPTIONS" ${ARGN}
    )

    if (${_arg_UNPARSED_ARGUMENTS})
        message(FATAL_ERROR "add_einsums_library had unparsed arguments")
    endif()

    update_cached_list(__EINSUMS_LIBRARIES "${name}")

    condition_info(_extra_text _arg_CONDITION)
    if (NOT _arg_CONDITION)
        set(_arg_CONDITION ON)
    endif()

    if (${_arg_CONDITION})
        set(_library_enabled ON)
    else()
        set(_library_enabled OFF)
    endif()

    if (DEFINED _arg_FEATURE_INFO)
        add_feature_info("Library ${name}" _library_enabled "${_extra_text}")
    endif()
    if (NOT _library_enabled)
        return()
    endif()

    set(library_type SHARED)
    if (_arg_STATIC OR (EINSUMS_STATIC_BUILD AND NOT _arg_SHARED))
        set(library_type STATIC)
    endif()
    if (_arg_OBJECT)
        set(library_type OBJECT)
    endif()

    add_library(${name} ${library_type})
    add_library(Einsums::${name} ALIAS ${name})

    # Disable CMake automatically defining ${name}_EXPORTS
    set_target_properties(${name}
        PROPERTIES
            DEFINE_SYMBOL ""
    )

    if (${name} MATCHES "^[^0-9-]+$")
        if (EINSUMS_STATIC_BUILD)
            set(export_symbol_suffix "STATIC_LIBRARY")
        else()
            set(export_symbol_suffix "LIBRARY")
        endif()
        string(TOUPPER "${name}_${export_symbol_suffix}" EXPORT_SYMBOL)
    endif()

    if (_arg_STATIC AND UNIX)
        set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    extend_einsums_target(${name}
        SOURCES_PREFIX ${_arg_SOURCES_PREFIX}
        SOURCES ${_arg_SOURCES}
        INCLUDES ${_arg_INCLUDES}
        PUBLIC_INCLUDES ${_arg_PUBLIC_INCLUDES}
        DEFINES ${default_defines_copy} ${_arg_DEFINES} ${TEST_DEFINES}
        PUBLIC_DEFINES ${_arg_PUBLIC_DEFINES}
        DEPENDS ${_arg_DEPENDS} ${IMPLICIT_DEPENDS}
        PUBLIC_DEPENDS ${_arg_PUBLIC_DEPENDS}
        OPTIONS ${_arg_OPTIONS}
        PUBLIC_OPTIONS ${_arg_PUBLIC_OPTIONS}
    )

    if (EINSUMS_STATIC_BUILD)
        extend_einsums_target(${name} PUBLIC_DEFINES ${EXPORT_SYMBOL})
    else()
        extend_einsums_target(${name} DEFINES ${EXPORT_SYMBOL})
    endif()

    # everything is different with SOURCES_PREFIX
    if (NOT _arg_SOURCES_PREFIX)
        get_filename_component(public_build_interface_dir "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
        file(RELATIVE_PATH include_dir_relative_path ${PROJECT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/..")
        target_include_directories(${name}
            PRIVATE
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            PUBLIC
                "$<BUILD_INTERFACE:${public_build_interface_dir}>"
                "$<INSTALL_INTERFACE:${EINSUMS_HEADER_INSTALL_PATH}/${include_dir_relative_path}>"
        )
    endif()

    set(_DESTINATION "lib")
    if (_arg_DESTINATION)
        set(_DESTINATION "${_arg_DESTINATION")
    endif()

    einsums_output_binary_dir(_output_binary_dir)
    string(REGEX MATCH "^[0-9]*" EINSUMS_VERSION_MAJOR ${EINSUMS_VERSION})
    set_target_properties(${name} PROPERTIES
        LINK_DEPENDS_NO_SHARED          ON
        SOURCES_DIR                     "${CMAKE_CURRENT_SOURCE_DIR}"
        VERSION                         "${EINSUMS_VERSION}"
        SOVERSION                       "${EINSUMS_VERSION_MAJOR}"
        MACHO_CURRENT_VERSION           ${EINSUMS_VERSION}
        MACHO_COMPATIBILITY_VERSION     ${EINSUMS_VERSION_COMPAT}
        CXX_EXTENSIONS                  OFF
        C_VISIBILITY_PRESET             hidden
        CXX_VISIBILITY_PRESET           hidden
        VISIBILITY_INLINES_HIDDEN       ON
        BUILD_RPATH                     "${_LIB_RPATH};${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH                   "${_LIB_RPATH};${CMAKE_INSTALL_RPATH}"
        RUNTIME_OUTPUT_DIRECTORY        "${_output_binary_dir}/${_DESTINATION}"
        LIBRARY_OUTPUT_DIRECTORY        "${_output_binary_dir}/${EINSUMS_LIBRARY_PATH}"
        ARCHIVE_OUTPUT_DIRECTORY        "${_output_binary_dir}/${EINSUMS_LIBRARY_ARCHIVE_PATH}"
        ${_arg_PROPERTIES}
    )

    unset(NAMELINK_OPTION)
    if (library_type STREQUAL "SHARED")
        set(NAMELINK_OPTION NAMELINK_SKIP)
        einsums_add_link_flags_no_undefined(${name})
    endif()

    unset(COMPONENT_OPTION)
    if (_arg_COMPONENT)
        set(COMPONENT_OPTION "COMPONENT" "${_arg_COMPONENT}")
    endif()

#    if (NOT EINSUMS_STATIC_BUILD OR _arg_SHARED)
#        install(TARGETS ${name}
#            EXPORT Einsums
#            RUNTIME
#                DESTINATION "${_DESTINATION}"
#                ${COMPONENT_OPTION}
#                OPTIONAL
#            LIBRARY
#                DESTINATION "${EINSUMS_LIBRARY_PATH}"
#                ${NAMELINK_OPTION}
#                ${COMPONENT_OPTION}
#                OPTIONAL
#            OBJECTS
#                DESTINATION "${EINSUMS_LIBRARY_PATH}"
#                COMPONENT Devel EXCLUDE_FROM_ALL
#            ARCHIVE
#                DESTINATION "${EINSUMS_LIBRARY_ARCHIVE_PATH}"
#                COMPONENT Devel EXCLUDE_FROM_ALL
#                OPTIONAL
#        )
#    endif()
#
#    if (NAMELINK_OPTION AND NOT EINSUMS_STATIC_BUILD)
#        install(TARGETS ${name}
#            LIBRARY
#                DESTINATION "${EINSUMS_LIBRARY_PATH}"
#                NAMELINK_ONLY
#                COMPONENT Devel EXCLUDE_FROM_ALL
#                OPTIONAL
#        )
#    endif()
endfunction(add_einsums_library)

function(einsums_add_public_header header)
    if (NOT IS_ABSOLUTE ${header})
        set(header "${CMAKE_CURRENT_SOURCE_DIR}/${header}")
    endif()

    einsums_source_dir(_einsums_source_dir)
    get_filename_component(source_dir ${header} DIRECTORY)
    file(RELATIVE_PATH include_dir_relative_path ${_einsums_source_dir} ${source_dir})

    #install(
    #    FILES ${header}
    #    DESTINATION "${EINSUMS_HEADER_INSTALL_PATH}/${include_dir_relative_path}"
    #    COMPONENT Devel EXCLUDE_FROM_ALL
    #)
endfunction()
