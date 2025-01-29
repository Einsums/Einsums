//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/StringUtil/FromString.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <iostream>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <string>

namespace einsums::detail {

EINSUMS_DETAIL_DEFINE_SPDLOG(einsums, warn)

spdlog::level::level_enum get_spdlog_level(std::string const &env) {
    try {
        return static_cast<spdlog::level::level_enum>(from_string<std::underlying_type_t<spdlog::level::level_enum>>(env));
    } catch (bad_lexical_cast const &) {
        fmt::print(std::cerr,
                   "Einsums given invalid log level: \"{}\". Using default level instead {} (warn). "
                   "Valid values are {} (trace) to {} (off).\n",
                   env, SPDLOG_LEVEL_WARN, SPDLOG_LEVEL_TRACE, SPDLOG_LEVEL_OFF);
        return spdlog::level::warn;
    }
}

std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const &env) {
    // In the future it might be useful to include a tcp sink option.
    // Could be useful when we are doing MPI/distributed development.
    if (env.empty()) {
        fmt::print(std::cerr, "Einsums given empty log destination. Using default instead (cerr).\n");
    } else if (env == "cout") {
        return std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    } else if (env == "cerr") {
        return std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    }
    return std::make_shared<spdlog::sinks::basic_file_sink_mt>(env);
}

} // namespace einsums::detail