//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/ConfigStrings.hpp>
#include <Einsums/Config/Version.hpp>
#include <Einsums/Version.hpp>

#include <fmt/format.h>

#include <sstream>
#include <string>

#define VERSION_MAJOR_STR EINSUMS_PP_STRINGIFY(EINSUMS_VERSION_MAJOR)
#define VERSION_MINOR_STR EINSUMS_PP_STRINGIFY(EINSUMS_VERSION_MINOR)
#define VERSION_PATCH_STR EINSUMS_PP_STRINGIFY(EINSUMS_VERSION_PATCH)

///////////////////////////////////////////////////////////////////////////////
namespace einsums {

std::string full_version_as_string() {
    // Do it this way so that it doesn't call a constexpr function. Windows struggles with constexpr strings it seems.
    return std::string{VERSION_MAJOR_STR "." VERSION_MINOR_STR "." VERSION_PATCH_STR};
}

std::string full_build_string() {
    std::ostringstream strm;
    strm << "{config}:\n"
         << configuration_string() << "{version}: " << build_string() << "\n"
         << "{build-type}: " << build_type() << "\n"
         << "{date}: " << build_date_time() << "\n";

    return strm.str();
}

///////////////////////////////////////////////////////////////////////////
std::string configuration_string() {
    std::ostringstream strm;

    strm << "Einsums:\n";

    char const *const *p = einsums::config_strings;
    while (*p)
        strm << "  " << *p++ << "\n";
    strm << "\n";

    return strm.str();
}

std::string build_string() {
    // This needs to be done like this so that the build string isn't evaluated as constexpr.
    // The Windows CRT struggles with copying constexpr strings it seems.

    char git_string[] = EINSUMS_HAVE_GIT_COMMIT;

    if (std::strlen(git_string) >= 10) {
        git_string[10] = '\0';
    }

    return std::string{"v" VERSION_MAJOR_STR "." VERSION_MINOR_STR "." VERSION_PATCH_STR EINSUMS_VERSION_TAG ", Git: "} + git_string;
}

std::string complete_version() {
    std::string out;

    auto runtime_format = fmt::runtime("Version:\n"
                                       "  Einsums: {}\n"
                                       "\n"
                                       "Build:\n"
                                       "  Type: {}\n"
                                       "  Date: {}\n");

    size_t out_size = fmt::formatted_size(runtime_format, build_string(), build_type(), build_date_time());

    out.resize(out_size);

    fmt::format_to(out.begin(), runtime_format, build_string(), build_type(), build_date_time());
    return out;
}

std::string build_date_time() {
    return std::string(__DATE__) + " " + __TIME__;
}

} // namespace einsums
