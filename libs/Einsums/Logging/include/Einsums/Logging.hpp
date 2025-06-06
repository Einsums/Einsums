//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Logging/Defines.hpp>

#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace einsums::detail {

#define EINSUMS_DETAIL_DECLARE_SPDLOG(name) spdlog::logger &get_##name##_logger() noexcept;

#define EINSUMS_DETAIL_DEFINE_SPDLOG(name, loglevel)                                                                                       \
    spdlog::logger &get_##name##_logger() noexcept {                                                                                       \
        static auto logger = []() {                                                                                                        \
            auto logger = std::make_shared<spdlog::logger>(#name, std::make_shared<spdlog::sinks::stderr_color_sink_mt>());                \
            logger->set_level(spdlog::level::loglevel);                                                                                    \
            return logger;                                                                                                                 \
        }();                                                                                                                               \
        static auto &logger_ref = *logger;                                                                                                 \
        return logger_ref;                                                                                                                 \
    }

#define EINSUMS_DETAIL_SPDLOG(name, loglevel, ...)                                                                                         \
    if (::einsums::detail::get_##name##_logger().level() <= loglevel) {                                                                    \
        ::einsums::detail::get_##name##_logger().log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},                              \
                                                     static_cast<::spdlog::level::level_enum>(loglevel), __VA_ARGS__);                     \
    }

#define EINSUMS_DETAIL_SPDLOG_ENABLED(name, loglevel) (::einsums::detail::get_##name##_logger().level() <= loglevel)

#define EINSUMS_LOG(loglevel, ...)    EINSUMS_DETAIL_SPDLOG(einsums, loglevel, __VA_ARGS__)
#define EINSUMS_LOG_ENABLED(loglevel) EINSUMS_DETAIL_SPDLOG_ENABLED(einsums, loglevel)

// Only EINSUMS_LOG_TRACE and EINSUMS_LOG_DEBUG can be disabled during compile time.
#if EINSUMS_ACTIVE_LOG_LEVEL <= 0
#    define EINSUMS_LOG_TRACE(...) EINSUMS_LOG(SPDLOG_LEVEL_TRACE, __VA_ARGS__)
#else
#    define EINSUMS_LOG_TRACE(...)
#endif

#if EINSUMS_ACTIVE_LOG_LEVEL <= 1
#    define EINSUMS_LOG_DEBUG(...) EINSUMS_LOG(SPDLOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#    define EINSUMS_LOG_DEBUG(...)
#endif

// These logging macros cannot be disabled at compile-time, but they can be disabled at runtime
#define EINSUMS_LOG_INFO(...)     EINSUMS_LOG(SPDLOG_LEVEL_INFO, __VA_ARGS__)
#define EINSUMS_LOG_WARN(...)     EINSUMS_LOG(SPDLOG_LEVEL_WARN, __VA_ARGS__)
#define EINSUMS_LOG_ERROR(...)    EINSUMS_LOG(SPDLOG_LEVEL_ERROR, __VA_ARGS__)
#define EINSUMS_LOG_CRITICAL(...) EINSUMS_LOG(SPDLOG_LEVEL_CRITICAL, __VA_ARGS__)

EINSUMS_EXPORT spdlog::level::level_enum get_spdlog_level(std::string const &env);
EINSUMS_EXPORT std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const &env);
EINSUMS_EXPORT                                      EINSUMS_DETAIL_DECLARE_SPDLOG(einsums)

} // namespace einsums::detail