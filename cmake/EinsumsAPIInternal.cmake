include(CheckLinkerFlag)
include(FeatureSummary)

#
# Setup path handling
#
include(GNUInstallDirs)

if (UNIX)
    set(_EINSUMS_LIBRARY_BASE_PATH "${CMAKE_INSTALL_LIBDIR}")
    set(_EINSUMS_LIBRARY_PATH "${_EINSUMS_LIBRARY_BASE_PATH}/einsums")
    set(_EINSUMS_LIBEXEC_PATH "${CMAKE_INSTALL_LIBEXECDIR}/einsums")
    set(_EINSUMS_DATA_PATH "${CMAKE_INSTALL_DATAROOTDIR}/einsums")
    set(_EINSUMS_DOC_PATH "${CMAKE_INSTALL_DATAROOTDIR}/doc/einsums")
    set(_EINSUMS_BIN_PATH "${CMAKE_INSTALL_BINDIR}")
    set(_EINSUMS_LIBRARY_ARCHIVE_PATH "${_EINSUMS_LIBRARY_PATH}")

    set(_EINSUMS_HEADER_INSTALL_PATH "include/einsums")
    set(_EINSUMS_CMAKE_INSTALL_PATH "${_EINSUMS_LIBRARY_BASE_PATH}/cmake")
elseif(WIN32)
    set(_EINSUMS_LIBRARY_BASE_PATH "lib")
    set(_EINSUMS_LIBRARY_PATH "${_EINSUMS_LIBRARY_BASE_PATH}/einsums")
    set(_EINSUMS_LIBEXEC_PATH "bin")
    set(_EINSUMS_DATA_PATH "share/einsums")
    set(_EINSUMS_DOC_PATH "share/doc/einsums")
    set(_EINSUMS_BIN_PATH "bin")
    set(_EINSUMS_LIBRARY_ARCHIVE_PATH "${_EINSUMS_BIN_PATH}")

    set(_EINSUMS_HEADER_INSTALL_PATH "include/einsums")
    set(_EINSUMS_CMAKE_INSTALL_PATH "lib/cmake")
endif ()

if (APPLE)
    set(_RPATH_BASE "@executable_path")
    set(_LIB_RPATH "@loader_path")
elseif (WIN32)
    set(_RPATH_BASE "")
    set(_LIB_RPATH "")
else()
    set(_RPATH_BASE "\$ORIGIN")
    set(_LIB_RPATH "\$ORIGIN")
endif ()

function(update_cached_list name value)
    set(_tmp_list "${${name}}")
    list(APPEND _tmp_list "${value}")
    set("${name}" "${_tmp_list}" CACHE INTERNAL "*** Internal ***")
endfunction()

function(set_public_headers target sources)
    foreach(source IN LISTS sources)
        if (source MATCHES "\.h$|\.hpp$")
            einsums_add_public_header(${source})
        endif()
    endforeach()
endfunction()


function(set_public_includes target includes)
    foreach(inc_dir IN LISTS includes)
        if (NOT IS_ABSOLUTE ${inc_dir})
            set(inc_dir "${CMAKE_CURRENT_SOURCE_DIR}/${inc_dir}")
        endif()
        file(RELATIVE_PATH include_dir_relative_path ${PROJECT_SOURCE_DIR} ${inc_dir})
        target_include_directories(${target} PUBLIC
            $<BUILD_INTERFACE:${inc_dir}>
            #$<INSTALL_INTERFACE:${_EINSUMS_HEADER_INSTALL_PATH}/${include_dir_relative_path}>
        )
    endforeach()
endfunction()

function(check_einsums_disabled_targets target_name dependent_targets)
    foreach(dependency IN LISTS ${dependent_targets})
        foreach(type PLUGIN LIBRARY)
            string(TOUPPER "BUILD_${type}_${dependency}" build_target)
            if (DEFINED ${build_target} AND NOT ${build_target})
                message(SEND_ERROR "Target ${name} depends on ${dependency} which was disabled via ${build_target} set to ${${build_target}}")
            endif()
        endforeach()
    endforeach()
endfunction()

function(add_einsums_depends target_name)
    cmake_parse_arguments(_arg "" "" "PRIVATE;PUBLIC" ${ARGN})
    if (${_arg_UNPARSED_ARGUMENTS})
        message(FATAL_ERROR "add_einsums_depends had unparsed arguments")
    endif()

    check_einsums_disabled_targets(${target_name} _arg_PRIVATE)
    check_einsums_disabled_targets(${target_name} _arg_PUBLIC)

    set(depends "${_arg_PRIVATE}")
    set(public_depends "${_arg_PUBLIC}")

    get_target_property(target_type ${target_name} TYPE)
    if (NOT target_type STREQUAL "OBJECT_LIBRARY")
        target_link_libraries(${target_name} PRIVATE ${depends} PUBLIC ${public_depends})
    else()
        list(APPEND object_lib_depends ${depends})
        list(APPEND object_public_depends ${public_depends})
    endif()

    foreach(obj_lib IN LISTS object_lib_depends)
        target_compile_options(${target_name} PRIVATE $<TARGET_PROPERTY:${obj_lib},INTERFACE_COMPILE_OPTIONS>)
        target_compile_definitions(${target_name} PRIVATE $<TARGET_PROPERTY:${obj_lib},INTERFACE_COMPILE_DEFINITIONS>)
        target_include_directories(${target_name} PRIVATE $<TARGET_PROPERTY:${obj_lib},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()
    foreach(obj_lib IN LISTS object_public_depends)
        target_compile_options(${target_name} PUBLIC $<TARGET_PROPERTY:${obj_lib},INTERFACE_COMPILE_OPTIONS>)
        target_compile_definitions(${target_name} PUBLIC $<TARGET_PROPERTY:${obj_lib},INTERFACE_COMPILE_DEFINITIONS>)
        target_include_directories(${target_name} PUBLIC $<TARGET_PROPERTY:${obj_lib},INTERFACE_INCLUDE_DIRECTORIES>)
   endforeach()
endfunction()

function(condition_info varName condition)
  if (NOT ${condition})
    set(${varName} "" PARENT_SCOPE)
  else()
    string(REPLACE ";" " " _contents "${${condition}}")
    set(${varName} "with CONDITION ${_contents}" PARENT_SCOPE)
  endif()
endfunction()

function(einsums_add_link_flags_no_undefined target)
    # needs CheckLinkerFlags
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set(no_undefined_flag "-Wl,--no-undefined")
        check_linker_flag(CXX ${no_undefined_flag} EINSUMS_LINKER_SUPPORTS_NO_UNDEFINED)
    if (NOT EINSUMS_LINKER_SUPPORTS_NO_UNDEFINED)
        set(no_undefined_flag "-Wl,-undefined,error")
        check_linker_flag(CXX ${no_undefined_flag} EINSUMS_LINKER_SUPPORTS_UNDEFINED_ERROR)
        if (NOT EINSUMS_LINKER_SUPPORTS_UNDEFINED_ERROR)
            return()
        endif()
    endif()
    target_link_options("${target}" PRIVATE "${no_undefined_flag}")
    endif()
endfunction()

function(condition_info varName condition)
    if (NOT ${condition})
        set(${varName} "" PARENT_SCOPE)
    else()
        string(REPLACE ";" " " _contents "${${condition}}")
        set(${varName} "with CONDITION ${_contents}" PARENT_SCOPE)
    endif()
endfunction()

function(extend_einsums_target target_name)
    cmake_parse_arguments(_arg
        ""
        "SOURCES_PREFIX;SOURCES_PREFIX_FROM_TARGET;FEATURE_INFO"
        "CONDITION;DEPENDS;PUBLIC_DEPENDS;DEFINES;PUBLIC_DEFINES;INCLUDES;PUBLIC_INCLUDES;SOURCES;PROPERTIES;OPTIONS;PUBLIC_OPTIONS"
        ${ARGN}
    )

    if (${_arg_UNPARSED_ARGUMENTS})
        message(FATAL_ERROR "extend_einsums_target had unparsed arguments")
    endif()

    condition_info(_extra_text _arg_CONDITION)
    if (NOT _arg_CONDITION)
        set(_arg_CONDITION ON)
    endif()
    if (${_arg_CONDITION})
        set(_feature_enabled ON)
    else()
        set(_feature_enabled OFF)
    endif()
    if (_arg_FEATURE_INFO)
        add_feature_info(${_arg_FEATURE_INFO} _feature_enabled "${_extra_text}")
    endif()
    if (NOT _feature_enabled)
        return()
    endif()

    if (_arg_SOURCES_PREFIX_FROM_TARGET)
        if (NOT TARGET ${_arg_SOURCES_PREFIX_FROM_TARGET})
            return()
        else()
            get_target_property(_arg_SOURCES_PREFIX ${_arg_SOURCES_PREFIX_FROM_TARGET} SOURCES_DIR)
        endif()
    endif()

    add_einsums_depends(${target_name}
        PRIVATE ${_arg_DEPENDS}
        PUBLIC ${_arg_PUBLIC_DEPENDS}
    )
    target_compile_definitions(${target_name}
        PRIVATE ${_arg_DEFINES}
        PUBLIC ${_arg_PUBLIC_DEFINES}
    )
    target_compile_options(${target_name}
        PRIVATE ${_arg_OPTIONS}
        PUBLIC ${_arg_PUBLIC_OPTIONS}
    )
    target_include_directories(${target_name} PRIVATE ${_arg_INCLUDES})

    set_public_includes(${target_name} "${_arg_PUBLIC_INCLUDES}")

    if (_arg_SOURCES_PREFIX)
        foreach(source IN LISTS _arg_SOURCES)
            list(APPEND prefixed_sources "${_arg_SOURCES_PREFIX}/${source}")
        endforeach()

        if (NOT IS_ABSOLUTE ${_arg_SOURCES_PREFIX})
            set(_arg_SOURCES_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/${_arg_SOURCES_PREFIX}")
        endif()
        target_include_directories(${target_name} PRIVATE $<BUILD_INTERFACE:${_arg_SOURCES_PREFIX}>)

        set(_arg_SOURCES ${prefixed_sources})
    endif()
    target_sources(${target_name} PRIVATE ${_arg_SOURCES})

    set_public_headers(${target_name} "${_arg_SOURCES}")

    if (_arg_PROPERTIES)
        set_target_properties(${target_name} PROPERTIES ${_arg_PROPERTIES})
    endif()
endfunction()
