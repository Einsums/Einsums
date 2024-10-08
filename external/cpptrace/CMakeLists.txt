if (NOT WIN32)
    include(FetchContent)
    fetchcontent_declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG v0.6.3 # <HASH or TAG>
        FIND_PACKAGE_ARGS 0.6.3
    )
    fetchcontent_makeavailable(cpptrace)
endif ()
