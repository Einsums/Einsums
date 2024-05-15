include(FetchContent)

# <<<  general notes  >>>

# * URL preferred over GIT_REPOSITORY+GIT_TAG
# * commit or permanent tag preferred over branch name for traceability
# * to get the URL_HASH, run `curl -sL <URL> | openssl sha256`
# * to use local source, put path in URL: `URL "/psi/gits/h5cpp"`
# * generally, do all FetchContent_Declare before any FetchContent_MakeAvailable
#   to allow for negotiation for any shared deps-of-deps. in practice,
#   we MakeAvailable hptt in its own dir to suppress install and h5cpp in its
#   own dir to collect build and install commands.

if(EINSUMS_ENABLE_TESTING)
    add_subdirectory(catch2)
endif()

add_subdirectory(fmt)

add_subdirectory(rangev3)

find_package(ZLIB REQUIRED)
find_package(TargetHDF5 REQUIRED)
# promote Mm version's scope so that it can be baked into EinsumsConfig
set(TargetHDF5_VERSION_Mm ${TargetHDF5_VERSION_Mm} PARENT_SCOPE)

add_subdirectory(h5cpp)

if(EINSUMS_USE_HPTT)
    # EXCLUDE_FROM_ALL to suppress install
    add_subdirectory(hptt EXCLUDE_FROM_ALL)

    if(EINSUMS_BUILD_HIP)
        add_subdirectory(librett EXCLUDE_FROM_ALL)
    endif()
endif()

if(EINSUMS_ENABLE_TESTING)
    FetchContent_MakeAvailable(Catch2)
endif()

FetchContent_MakeAvailable(fmt)

FetchContent_MakeAvailable(range-v3)
