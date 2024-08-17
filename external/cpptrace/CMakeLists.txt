include(FetchContent)
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG v0.6.3 # <HASH or TAG>
        FIND_PACKAGE_ARGS 0.6.3
)
FetchContent_MakeAvailable(cpptrace)
