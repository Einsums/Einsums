//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/Detail/InitLogging.hpp>
#include <Einsums/RuntimeConfiguration/RuntimeConfiguration.hpp>

#include <fmt/format.h>

#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>
#include <string>

namespace einsums::detail {

namespace {
/*
 * The desired output is "tid/description". Modern operating systems
 * allow you to set a description on a thread that can be accessed.
 * This could be useful in the future when we introduce thread pools
 * The logger can output the thread description to aid in debugging.
 */
void spdlog_format_thread_id(int pid, spdlog::details::log_msg const &, std::tm const &, spdlog::memory_buf_t &dest) {
    dest.append(fmt::format("{}/----", pid));
}
} // namespace

struct ThreadIdFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
        spdlog_format_thread_id(getpid(), msg, tm_time, dest);
    }

    std::unique_ptr<custom_flag_formatter> clone() const override { return spdlog::details::make_unique<ThreadIdFormatterFlag>(); }
};

struct ParentThreadIdFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
#if defined(EINSUMS_WINDOWS)
        /// @todo There is a way to get the parent pid on Windows. Just don't want to do it now.
        spdlog_format_thread_id(0, msg, tm_time, dest);
#else
        spdlog_format_thread_id(getppid(), msg, tm_time, dest);
#endif
    }

    std::unique_ptr<custom_flag_formatter> clone() const override { return spdlog::details::make_unique<ParentThreadIdFormatterFlag>(); }
};

/*
 * The desired output is "hostname" or eventually when MPI is added
 * "hostname/rank".
 */
struct HostnameFormatterFlag : spdlog::custom_flag_formatter {
    void format(spdlog::details::log_msg const &msg, std::tm const &tm_time, spdlog::memory_buf_t &dest) override {
        dest.append(std::string("localhost"));
    }

    std::unique_ptr<custom_flag_formatter> clone() const override { return spdlog::details::make_unique<HostnameFormatterFlag>(); }
};

void init_logging(RuntimeConfiguration &config) {
    auto &global_config = GlobalConfigMap::get_singleton();
    // Set log destination
    auto &sinks = get_einsums_logger().sinks();
    sinks.clear();
    sinks.push_back(get_spdlog_sink(global_config.get_string("log-destination")));

    // Set log pattern
    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<ThreadIdFormatterFlag>('k');
    formatter->add_flag<ParentThreadIdFormatterFlag>('q');
    formatter->add_flag<HostnameFormatterFlag>('j');
    formatter->set_pattern(global_config.get_string("log-format"));
    get_einsums_logger().set_formatter(std::move(formatter));

    // Set log level
    get_einsums_logger().set_level(static_cast<spdlog::level::level_enum>(global_config.get_int("log-level")));

    EINSUMS_LOG_INFO("logging submodule has been initialized");
    EINSUMS_LOG_INFO("log level: {} (0=TRACE,1=DEBUG,2=INFO,3=WARN,4=ERROR,5=CRITICAL)", global_config.get_int("log-level"));
    // EINSUMS_LOG_DEBUG("test debug");
    // EINSUMS_LOG_TRACE("test trace");
    // EINSUMS_LOG_INFO("test info");
    // EINSUMS_LOG_WARN("test warn");
    // EINSUMS_LOG_ERROR("test error");
    // EINSUMS_LOG_CRITICAL("test critical");
}

} // namespace einsums::detail