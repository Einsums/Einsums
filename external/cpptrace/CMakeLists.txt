include(FetchContent)
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG v0.6.3 # <HASH or TAG>
)
FetchContent_MakeAvailable(cpptrace)

# On windows copy cpptrace.dll to the same directory as the executable for your_target
if (WIN32)
    add_custom_command(
            TARGET your_target POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cpptrace::cpptrace>
            $<TARGET_FILE_DIR:your_target>
    )
endif ()