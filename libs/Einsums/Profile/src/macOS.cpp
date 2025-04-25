//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/Detail/PerformanceCounter.hpp>

#include <dlfcn.h>          // for dlopen() dlclose(), dlsym()
#include <mach/mach_time.h> // for mach_absolute_time()
#include <sys/kdebug.h>     // for kdebug trace decode
#include <sys/sysctl.h>     // for sysctl()
#include <unistd.h>         // for usleep()

typedef float    f32;
typedef double   f64;
typedef int8_t   i8;
typedef uint8_t  u8;
typedef int16_t  i16;
typedef uint16_t u16;
typedef int32_t  i32;
typedef uint32_t u32;
typedef int64_t  i64;
typedef uint64_t u64;
typedef size_t   usize;

// -----------------------------------------------------------------------------
// <kperf.framework> header (reverse engineered)
// This framework wraps some sysctl calls to communicate with the kpc in kernel.
// Most functions requires root privileges, or process is "blessed".
// -----------------------------------------------------------------------------

// Cross-platform class constants.
#define KPC_CLASS_FIXED        (0)
#define KPC_CLASS_CONFIGURABLE (1)
#define KPC_CLASS_POWER        (2)
#define KPC_CLASS_RAWPMU       (3)

// Cross-platform class mask constants.
#define KPC_CLASS_FIXED_MASK        (1u << KPC_CLASS_FIXED)        // 1
#define KPC_CLASS_CONFIGURABLE_MASK (1u << KPC_CLASS_CONFIGURABLE) // 2
#define KPC_CLASS_POWER_MASK        (1u << KPC_CLASS_POWER)        // 4
#define KPC_CLASS_RAWPMU_MASK       (1u << KPC_CLASS_RAWPMU)       // 8

// PMU version constants.
#define KPC_PMU_ERROR     (0) // Error
#define KPC_PMU_INTEL_V3  (1) // Intel
#define KPC_PMU_ARM_APPLE (2) // ARM64
#define KPC_PMU_INTEL_V2  (3) // Old Intel
#define KPC_PMU_ARM_V2    (4) // Old ARM

// The maximum number of counters we could read from every class in one go.
// ARMV7: FIXED: 1, CONFIGURABLE: 4
// ARM32: FIXED: 2, CONFIGURABLE: 6
// ARM64: FIXED: 2, CONFIGURABLE: CORE_NCTRS - FIXED (6 or 8)
// x86: 32
#define KPC_MAX_COUNTERS 32

// Bits for defining what to do on an action.
// Defined in https://github.com/apple/darwin-xnu/blob/main/osfmk/kperf/action.h
#define KPERF_SAMPLER_TH_INFO       (1U << 0)
#define KPERF_SAMPLER_TH_SNAPSHOT   (1U << 1)
#define KPERF_SAMPLER_KSTACK        (1U << 2)
#define KPERF_SAMPLER_USTACK        (1U << 3)
#define KPERF_SAMPLER_PMC_THREAD    (1U << 4)
#define KPERF_SAMPLER_PMC_CPU       (1U << 5)
#define KPERF_SAMPLER_PMC_CONFIG    (1U << 6)
#define KPERF_SAMPLER_MEMINFO       (1U << 7)
#define KPERF_SAMPLER_TH_SCHEDULING (1U << 8)
#define KPERF_SAMPLER_TH_DISPATCH   (1U << 9)
#define KPERF_SAMPLER_TK_SNAPSHOT   (1U << 10)
#define KPERF_SAMPLER_SYS_MEM       (1U << 11)
#define KPERF_SAMPLER_TH_INSCYC     (1U << 12)
#define KPERF_SAMPLER_TK_INFO       (1U << 13)

// Maximum number of kperf action ids.
#define KPERF_ACTION_MAX (32)

// Maximum number of kperf timer ids.
#define KPERF_TIMER_MAX (8)

// x86/arm config registers are 64-bit
typedef u64 kpc_config_t;

/// Print current CPU identification string to the buffer (same as snprintf),
/// such as "cpu_7_8_10b282dc_46". This string can be used to locate the PMC
/// database in /usr/share/kpep.
/// @return string's length, or negative value if error occurs.
/// @note This method does not requires root privileges.
/// @details sysctl get(hw.cputype), get(hw.cpusubtype),
///                 get(hw.cpufamily), get(machdep.cpu.model)
static int (*kpc_cpu_string)(char *buf, usize buf_size);

/// Get the version of KPC that's being run.
/// @return See `PMU version constants` above.
/// @details sysctl get(kpc.pmu_version)
static u32 (*kpc_pmu_version)();

/// Get running PMC classes.
/// @return See `class mask constants` above,
///         0 if error occurs or no class is set.
/// @details sysctl get(kpc.counting)
static u32 (*kpc_get_counting)();

/// Set PMC classes to enable counting.
/// @param classes See `class mask constants` above, set 0 to shutdown counting.
/// @return 0 for success.
/// @details sysctl set(kpc.counting)
static int (*kpc_set_counting)(u32 classes);

/// Get running PMC classes for current thread.
/// @return See `class mask constants` above,
///         0 if error occurs or no class is set.
/// @details sysctl get(kpc.thread_counting)
static u32 (*kpc_get_thread_counting)();

/// Set PMC classes to enable counting for current thread.
/// @param classes See `class mask constants` above, set 0 to shutdown counting.
/// @return 0 for success.
/// @details sysctl set(kpc.thread_counting)
static int (*kpc_set_thread_counting)(u32 classes);

/// Get how many config registers there are for a given mask.
/// For example: Intel may returns 1 for `KPC_CLASS_FIXED_MASK`,
///                        returns 4 for `KPC_CLASS_CONFIGURABLE_MASK`.
/// @param classes See `class mask constants` above.
/// @return 0 if error occurs or no class is set.
/// @note This method does not require root privileges.
/// @details sysctl get(kpc.config_count)
static u32 (*kpc_get_config_count)(u32 classes);

/// Get config registers.
/// @param classes see `class mask constants` above.
/// @param config Config buffer to receive values, should not smaller than
///               kpc_get_config_count(classes) * sizeof(kpc_config_t).
/// @return 0 for success.
/// @details sysctl get(kpc.config_count), get(kpc.config)
static int (*kpc_get_config)(u32 classes, kpc_config_t *config);

/// Set config registers.
/// @param classes see `class mask constants` above.
/// @param config Config buffer, should not smaller than
///               kpc_get_config_count(classes) * sizeof(kpc_config_t).
/// @return 0 for success.
/// @details sysctl get(kpc.config_count), set(kpc.config)
static int (*kpc_set_config)(u32 classes, kpc_config_t *config);

/// Get how many counters there are for a given mask.
/// For example: Intel may returns 3 for `KPC_CLASS_FIXED_MASK`,
///                        returns 4 for `KPC_CLASS_CONFIGURABLE_MASK`.
/// @param classes See `class mask constants` above.
/// @note This method does not requires root privileges.
/// @details sysctl get(kpc.counter_count)
static u32 (*kpc_get_counter_count)(u32 classes);

/// Get counter accumulations.
/// If `all_cpus` is true, the buffer count should not smaller than
/// (cpu_count * counter_count). Otherwize, the buffer count should not smaller
/// than (counter_count).
/// @see kpc_get_counter_count(), kpc_cpu_count().
/// @param all_cpus true for all CPUs, false for current cpu.
/// @param classes See `class mask constants` above.
/// @param curcpu A pointer to receive current cpu id, can be NULL.
/// @param buf Buffer to receive counter's value.
/// @return 0 for success.
/// @details sysctl get(hw.ncpu), get(kpc.counter_count), get(kpc.counters)
static int (*kpc_get_cpu_counters)(bool all_cpus, u32 classes, int *curcpu, u64 *buf);

/// Get counter accumulations for current thread.
/// @param tid Thread id, should be 0.
/// @param buf_count The number of buf's elements (not bytes),
///                  should not smaller than kpc_get_counter_count().
/// @param buf Buffer to receive counter's value.
/// @return 0 for success.
/// @details sysctl get(kpc.thread_counters)
static int (*kpc_get_thread_counters)(u32 tid, u32 buf_count, u64 *buf);

/// Acquire/release the counters used by the Power Manager.
/// @param val 1:acquire, 0:release
/// @return 0 for success.
/// @details sysctl set(kpc.force_all_ctrs)
static int (*kpc_force_all_ctrs_set)(int val);

/// Get the state of all_ctrs.
/// @return 0 for success.
/// @details sysctl get(kpc.force_all_ctrs)
static int (*kpc_force_all_ctrs_get)(int *val_out);

/// Set number of actions, should be `KPERF_ACTION_MAX`.
/// @details sysctl set(kperf.action.count)
static int (*kperf_action_count_set)(u32 count);

/// Get number of actions.
/// @details sysctl get(kperf.action.count)
static int (*kperf_action_count_get)(u32 *count);

/// Set what to sample when a trigger fires an action, e.g. `KPERF_SAMPLER_PMC_CPU`.
/// @details sysctl set(kperf.action.samplers)
static int (*kperf_action_samplers_set)(u32 actionid, u32 sample);

/// Get what to sample when a trigger fires an action.
/// @details sysctl get(kperf.action.samplers)
static int (*kperf_action_samplers_get)(u32 actionid, u32 *sample);

/// Apply a task filter to the action, -1 to disable filter.
/// @details sysctl set(kperf.action.filter_by_task)
static int (*kperf_action_filter_set_by_task)(u32 actionid, i32 port);

/// Apply a pid filter to the action, -1 to disable filter.
/// @details sysctl set(kperf.action.filter_by_pid)
static int (*kperf_action_filter_set_by_pid)(u32 actionid, i32 pid);

/// Set number of time triggers, should be `KPERF_TIMER_MAX`.
/// @details sysctl set(kperf.timer.count)
static int (*kperf_timer_count_set)(u32 count);

/// Get number of time triggers.
/// @details sysctl get(kperf.timer.count)
static int (*kperf_timer_count_get)(u32 *count);

/// Set timer number and period.
/// @details sysctl set(kperf.timer.period)
static int (*kperf_timer_period_set)(u32 actionid, u64 tick);

/// Get timer number and period.
/// @details sysctl get(kperf.timer.period)
static int (*kperf_timer_period_get)(u32 actionid, u64 *tick);

/// Set timer number and actionid.
/// @details sysctl set(kperf.timer.action)
static int (*kperf_timer_action_set)(u32 actionid, u32 timerid);

/// Get timer number and actionid.
/// @details sysctl get(kperf.timer.action)
static int (*kperf_timer_action_get)(u32 actionid, u32 *timerid);

/// Set which timer ID does PET (Profile Every Thread).
/// @details sysctl set(kperf.timer.pet_timer)
static int (*kperf_timer_pet_set)(u32 timerid);

/// Get which timer ID does PET (Profile Every Thread).
/// @details sysctl get(kperf.timer.pet_timer)
static int (*kperf_timer_pet_get)(u32 *timerid);

/// Enable or disable sampling.
/// @details sysctl set(kperf.sampling)
static int (*kperf_sample_set)(u32 enabled);

/// Get is currently sampling.
/// @details sysctl get(kperf.sampling)
static int (*kperf_sample_get)(u32 *enabled);

/// Reset kperf: stop sampling, kdebug, timers and actions.
/// @return 0 for success.
static int (*kperf_reset)();

/// Nanoseconds to CPU ticks.
static u64 (*kperf_ns_to_ticks)(u64 ns);

/// CPU ticks to nanoseconds.
static u64 (*kperf_ticks_to_ns)(u64 ticks);

/// CPU ticks frequency (mach_absolute_time).
static u64 (*kperf_tick_frequency)();

/// Get lightweight PET mode (not in kperf.framework).
static int kperf_lightweight_pet_get(u32 *enabled) {
    if (!enabled)
        return -1;
    usize size = 4;
    return sysctlbyname("kperf.lightweight_pet", enabled, &size, nullptr, 0);
}

/// Set lightweight PET mode (not in kperf.framework).
static int kperf_lightweight_pet_set(u32 enabled) {
    return sysctlbyname("kperf.lightweight_pet", nullptr, nullptr, &enabled, 4);
}

// -----------------------------------------------------------------------------
// <kperfdata.framework> header (reverse engineered)
// This framework provides some functions to access the local CPU database.
// These functions do not require root privileges.
// -----------------------------------------------------------------------------

// KPEP CPU archtecture constants.
#define KPEP_ARCH_I386   0
#define KPEP_ARCH_X86_64 1
#define KPEP_ARCH_ARM    2
#define KPEP_ARCH_ARM64  3

/// KPEP event (size: 48/28 bytes on 64/32 bit OS)
typedef struct kpep_event {
    char const *name;        ///< Unique name of a event, such as "INST_RETIRED.ANY".
    char const *description; ///< Description for this event.
    char const *errata;      ///< Errata, currently NULL.
    char const *alias;       ///< Alias name, such as "Instructions", "Cycles".
    char const *fallback;    ///< Fallback event name for fixed counter.
    u32         mask;
    u8          number;
    u8          umask;
    u8          reserved;
    u8          is_fixed;
} kpep_event;

/// KPEP database (size: 144/80 bytes on 64/32 bit OS)
typedef struct kpep_db {
    char const  *name;            ///< Database name, such as "haswell".
    char const  *cpu_id;          ///< Plist name, such as "cpu_7_8_10b282dc".
    char const  *marketing_name;  ///< Marketing name, such as "Intel Haswell".
    void        *plist_data;      ///< Plist data (CFDataRef), currently NULL.
    void        *event_map;       ///< All events (CFDict<CFSTR(event_name), kpep_event *>).
    kpep_event  *event_arr;       ///< Event struct buffer (sizeof(kpep_event) * events_count).
    kpep_event **fixed_event_arr; ///< Fixed counter events (sizeof(kpep_event *) * fixed_counter_count)
    void        *alias_map;       ///< All aliases (CFDict<CFSTR(event_name), kpep_event *>).
    usize        reserved_1;
    usize        reserved_2;
    usize        reserved_3;
    usize        event_count; ///< All events count.
    usize        alias_count;
    usize        fixed_counter_count;
    usize        config_counter_count;
    usize        power_counter_count;
    u32          archtecture; ///< see `KPEP CPU archtecture constants` above.
    u32          fixed_counter_bits;
    u32          config_counter_bits;
    u32          power_counter_bits;
} kpep_db;

/// KPEP config (size: 80/44 bytes on 64/32 bit OS)
typedef struct kpep_config {
    kpep_db     *db;
    kpep_event **ev_arr;      ///< (sizeof(kpep_event *) * counter_count), init NULL
    usize       *ev_map;      ///< (sizeof(usize *) * counter_count), init 0
    usize       *ev_idx;      ///< (sizeof(usize *) * counter_count), init -1
    u32         *flags;       ///< (sizeof(u32 *) * counter_count), init 0
    u64         *kpc_periods; ///< (sizeof(u64 *) * counter_count), init 0
    usize        event_count; /// kpep_config_events_count()
    usize        counter_count;
    u32          classes; ///< See `class mask constants` above.
    u32          config_counter;
    u32          power_counter;
    u32          reserved;
} kpep_config;

/// Error code for kpep_config_xxx() and kpep_db_xxx() functions.
typedef enum {
    KPEP_CONFIG_ERROR_NONE                   = 0,
    KPEP_CONFIG_ERROR_INVALID_ARGUMENT       = 1,
    KPEP_CONFIG_ERROR_OUT_OF_MEMORY          = 2,
    KPEP_CONFIG_ERROR_IO                     = 3,
    KPEP_CONFIG_ERROR_BUFFER_TOO_SMALL       = 4,
    KPEP_CONFIG_ERROR_CUR_SYSTEM_UNKNOWN     = 5,
    KPEP_CONFIG_ERROR_DB_PATH_INVALID        = 6,
    KPEP_CONFIG_ERROR_DB_NOT_FOUND           = 7,
    KPEP_CONFIG_ERROR_DB_ARCH_UNSUPPORTED    = 8,
    KPEP_CONFIG_ERROR_DB_VERSION_UNSUPPORTED = 9,
    KPEP_CONFIG_ERROR_DB_CORRUPT             = 10,
    KPEP_CONFIG_ERROR_EVENT_NOT_FOUND        = 11,
    KPEP_CONFIG_ERROR_CONFLICTING_EVENTS     = 12,
    KPEP_CONFIG_ERROR_COUNTERS_NOT_FORCED    = 13,
    KPEP_CONFIG_ERROR_EVENT_UNAVAILABLE      = 14,
    KPEP_CONFIG_ERROR_ERRNO                  = 15,
    KPEP_CONFIG_ERROR_MAX
} kpep_config_error_code;

/// Error description for kpep_config_error_code.
static char const *kpep_config_error_names[KPEP_CONFIG_ERROR_MAX] = {"none",
                                                                     "invalid argument",
                                                                     "out of memory",
                                                                     "I/O",
                                                                     "buffer too small",
                                                                     "current system unknown",
                                                                     "database path invalid",
                                                                     "database not found",
                                                                     "database architecture unsupported",
                                                                     "database version unsupported",
                                                                     "database corrupt",
                                                                     "event not found",
                                                                     "conflicting events",
                                                                     "all counters must be forced",
                                                                     "event unavailable",
                                                                     "check errno"};

/// Error description.
static char const *kpep_config_error_desc(int code) {
    if (0 <= code && code < KPEP_CONFIG_ERROR_MAX) {
        return kpep_config_error_names[code];
    }
    return "unknown error";
}

/// Create a config.
/// @param db A kpep db, see kpep_db_create()
/// @param cfg_ptr A pointer to receive the new config.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_create)(kpep_db *db, kpep_config **cfg_ptr);

/// Free the config.
static void (*kpep_config_free)(kpep_config *cfg);

/// Add an event to config.
/// @param cfg The config.
/// @param ev_ptr A event pointer.
/// @param flag 0: all, 1: user space only
/// @param err Error bitmap pointer, can be NULL.
///            If return value is `CONFLICTING_EVENTS`, this bitmap contains
///            the conflicted event indices, e.g. "1 << 2" means index 2.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_add_event)(kpep_config *cfg, kpep_event **ev_ptr, u32 flag, u32 *err);

/// Remove event at index.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_remove_event)(kpep_config *cfg, usize idx);

/// Force all counters.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_force_counters)(kpep_config *cfg);

/// Get events count.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_events_count)(kpep_config *cfg, usize *count_ptr);

/// Get all event pointers.
/// @param buf A buffer to receive event pointers.
/// @param buf_size The buffer's size in bytes, should not smaller than
///                 kpep_config_events_count() * sizeof(void *).
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_events)(kpep_config *cfg, kpep_event **buf, usize buf_size);

/// Get kpc register configs.
/// @param buf A buffer to receive kpc register configs.
/// @param buf_size The buffer's size in bytes, should not smaller than
///                 kpep_config_kpc_count() * sizeof(kpc_config_t).
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_kpc)(kpep_config *cfg, kpc_config_t *buf, usize buf_size);

/// Get kpc register config count.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_kpc_count)(kpep_config *cfg, usize *count_ptr);

/// Get kpc classes.
/// @param classes_ptr See `class mask constants` above.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_kpc_classes)(kpep_config *cfg, u32 *classes_ptr);

/// Get the index mapping from event to counter.
/// @param buf A buffer to receive indexes.
/// @param buf_size The buffer's size in bytes, should not smaller than
///                 kpep_config_events_count() * sizeof(kpc_config_t).
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_config_kpc_map)(kpep_config *cfg, usize *buf, usize buf_size);

/// Open a kpep database file in "/usr/share/kpep/" or "/usr/local/share/kpep/".
/// @param name File name, for example "haswell", "cpu_100000c_1_92fb37c8".
///             Pass NULL for current CPU.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_create)(char const *name, kpep_db **db_ptr);

/// Free the kpep database.
static void (*kpep_db_free)(kpep_db *db);

/// Get the database's name.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_name)(kpep_db *db, char const **name);

/// Get the event alias count.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_aliases_count)(kpep_db *db, usize *count);

/// Get all alias.
/// @param buf A buffer to receive all alias strings.
/// @param buf_size The buffer's size in bytes,
///        should not smaller than kpep_db_aliases_count() * sizeof(void *).
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_aliases)(kpep_db *db, char const **buf, usize buf_size);

/// Get counters count for given classes.
/// @param classes 1: Fixed, 2: Configurable.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_counters_count)(kpep_db *db, u8 classes, usize *count);

/// Get all event count.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_events_count)(kpep_db *db, usize *count);

/// Get all events.
/// @param buf A buffer to receive all event pointers.
/// @param buf_size The buffer's size in bytes,
///        should not smaller than kpep_db_events_count() * sizeof(void *).
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_events)(kpep_db *db, kpep_event **buf, usize buf_size);

/// Get one event by name.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_db_event)(kpep_db *db, char const *name, kpep_event **ev_ptr);

/// Get event's name.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_event_name)(kpep_event *ev, char const **name_ptr);

/// Get event's alias.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_event_alias)(kpep_event *ev, char const **alias_ptr);

/// Get event's description.
/// @return kpep_config_error_code, 0 for success.
static int (*kpep_event_description)(kpep_event *ev, char const **str_ptr);

namespace einsums::profile::detail {

namespace {

struct Library {
    struct Symbol {
        char const *name;
        void      **impl;
    };

#define SymbolDef(name) {#name, (void **)&name}
    const std::vector<Symbol> symbols_kperf = {
        SymbolDef(kpc_pmu_version),
        SymbolDef(kpc_cpu_string),
        SymbolDef(kpc_set_counting),
        SymbolDef(kpc_get_counting),
        SymbolDef(kpc_set_thread_counting),
        SymbolDef(kpc_get_thread_counting),
        SymbolDef(kpc_get_config_count),
        SymbolDef(kpc_get_counter_count),
        SymbolDef(kpc_set_config),
        SymbolDef(kpc_get_config),
        SymbolDef(kpc_get_cpu_counters),
        SymbolDef(kpc_get_thread_counters),
        SymbolDef(kpc_force_all_ctrs_set),
        SymbolDef(kpc_force_all_ctrs_get),
        SymbolDef(kperf_action_count_set),
        SymbolDef(kperf_action_count_get),
        SymbolDef(kperf_action_samplers_set),
        SymbolDef(kperf_action_samplers_get),
        SymbolDef(kperf_action_filter_set_by_task),
        SymbolDef(kperf_action_filter_set_by_pid),
        SymbolDef(kperf_timer_count_set),
        SymbolDef(kperf_timer_count_get),
        SymbolDef(kperf_timer_period_set),
        SymbolDef(kperf_timer_period_get),
        SymbolDef(kperf_timer_action_set),
        SymbolDef(kperf_timer_action_get),
        SymbolDef(kperf_sample_set),
        SymbolDef(kperf_sample_get),
        SymbolDef(kperf_reset),
        SymbolDef(kperf_timer_pet_set),
        SymbolDef(kperf_timer_pet_get),
        SymbolDef(kperf_ns_to_ticks),
        SymbolDef(kperf_ticks_to_ns),
        SymbolDef(kperf_tick_frequency),
    };

    std::vector<Symbol> const symbols_kperfdata = {
        SymbolDef(kpep_config_create),
        SymbolDef(kpep_config_free),
        SymbolDef(kpep_config_add_event),
        SymbolDef(kpep_config_remove_event),
        SymbolDef(kpep_config_force_counters),
        SymbolDef(kpep_config_events_count),
        SymbolDef(kpep_config_events),
        SymbolDef(kpep_config_kpc),
        SymbolDef(kpep_config_kpc_count),
        SymbolDef(kpep_config_kpc_classes),
        SymbolDef(kpep_config_kpc_map),
        SymbolDef(kpep_db_create),
        SymbolDef(kpep_db_free),
        SymbolDef(kpep_db_name),
        SymbolDef(kpep_db_aliases_count),
        SymbolDef(kpep_db_aliases),
        SymbolDef(kpep_db_counters_count),
        SymbolDef(kpep_db_events_count),
        SymbolDef(kpep_db_events),
        SymbolDef(kpep_db_event),
        SymbolDef(kpep_event_name),
        SymbolDef(kpep_event_alias),
        SymbolDef(kpep_event_description),
    };

    static constexpr char const *library_path_kperf     = "/System/Library/PrivateFrameworks/kperf.framework/kperf";
    static constexpr char const *library_path_kperfdata = "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata";

    bool initialized{false};
    bool has_error{false};

    void *library_handle_kperf{nullptr};
    void *library_handle_kperfdata{nullptr};

    void deinitialize() {
        initialized = false;
        has_error   = false;
        if (library_handle_kperf) {
            dlclose(library_handle_kperf);
        }
        if (library_handle_kperfdata) {
            dlclose(library_handle_kperfdata);
        }
        library_handle_kperf     = nullptr;
        library_handle_kperfdata = nullptr;
        for (auto symbol : symbols_kperf) {
            *symbol.impl = nullptr;
        }
        for (auto symbol : symbols_kperfdata) {
            *symbol.impl = nullptr;
        }
    }

    bool initialize() {
#define return_error()                                                                                                                     \
    do {                                                                                                                                   \
        deinitialize();                                                                                                                    \
        initialized = true;                                                                                                                \
        has_error   = true;                                                                                                                \
        return false;                                                                                                                      \
    } while (false)

        if (initialized) {
            return !has_error;
        }

        // load dynamic library
        library_handle_kperf = dlopen(library_path_kperf, RTLD_LAZY);
        if (!library_handle_kperf) {
            EINSUMS_LOG_ERROR("Failed to load kperf.framework, message: {}", dlerror());
            return_error();
        }
        library_handle_kperfdata = dlopen(library_path_kperfdata, RTLD_LAZY);
        if (!library_handle_kperfdata) {
            EINSUMS_LOG_ERROR("Failed to load kperfdata.framework, message: {}", dlerror());
            return_error();
        }

        // load symbol address from dynamic library
        for (auto symbol : symbols_kperf) {
            *symbol.impl = dlsym(library_handle_kperf, symbol.name);
            if (!*symbol.impl) {
                EINSUMS_LOG_ERROR("Failed to load kperf function: {}, message: {}", symbol.name, dlerror());
                return_error();
            }
        }
        for (auto symbol : symbols_kperfdata) {
            *symbol.impl = dlsym(library_handle_kperfdata, symbol.name);
            if (!*symbol.impl) {
                EINSUMS_LOG_ERROR("Failed to load kperfdata function: {}, message: {}", symbol.name, dlerror());
                return_error();
            }
        }

        initialized = true;
        has_error   = false;
        return true;
#undef return_error
    }
};

} // namespace

struct PerformanceCounterMac : PerformanceCounter {

    static constexpr int EVENT_NAME_MAX = 8;
    struct EventAlias {
        char const *alias; // name for print
        char const *names[EVENT_NAME_MAX];
    };

    // Event names from /usr/share/keep/<name>.plist
    static constexpr EventAlias profile_events[] = {
        {"cycles",
         {
             "FIXED_CYCLES",            // Apple A7-A15
             "CPU_CLK_UNHALTED.THREAD", // Intel Core 1th-10th
             "CPU_CLK_UNHALTED.CORE",   // Intel Yonah, Merom
         }},
        {"instructions",
         {
             "FIXED_INSTRUCTIONS", // Apple A7-A15
             "INST_RETIRED.ANY"    // Intel Yonah, Merom, Core 1th-10th
         }},
        {"branches",
         {
             "INST_BRANCH",                  // Apple A7-A15
             "BR_INST_RETIRED.ALL_BRANCHES", // Intel Core 1th-10th
             "INST_RETIRED.ANY",             // Intel Yonah, Merom
         }},
        {"branch-misses",
         {
             "BRANCH_MISPRED_NONSPEC",       // Apple A7-A15, since iOS 15, macOS 12
             "BRANCH_MISPREDICT",            // Apple A7-A14
             "BR_MISP_RETIRED.ALL_BRANCHES", // Intel Core 2th-10th
             "BR_INST_RETIRED.MISPRED",      // Intel Yonah, Merom
         }},
    };
    static constexpr int num_counters = std::size(profile_events);

    std::array<kpc_config_t, KPC_MAX_COUNTERS> registers    = {};
    std::array<usize, KPC_MAX_COUNTERS>        counter_map  = {};
    std::array<uint64_t, KPC_MAX_COUNTERS>     counters     = {};
    std::array<uint64_t, KPC_MAX_COUNTERS>     start_values = {};
    std::array<uint64_t, KPC_MAX_COUNTERS>     delta_values = {};

    static kpep_event *get_event(kpep_db *db, EventAlias const *alias) {
        for (auto name : alias->names) {
            if (!name)
                break;
            kpep_event *ev = nullptr;
            if (kpep_db_event(db, name, &ev) == 0) {
                return ev;
            }
        }
        return nullptr;
    }

    PerformanceCounterMac() {
        // load dylib
        if (!_library.initialize()) {
            EINSUMS_LOG_ERROR("Failed to initialize library.");
            return;
        }

        // check permission
        int force_ctrs{0};
        if (kpc_force_all_ctrs_get(&force_ctrs)) {
            EINSUMS_LOG_ERROR("Permission denied. xnu/kpc requires root privileges.");
            return;
        }

        int ret;
        // load pmc db
        kpep_db *db{nullptr};
        if ((ret = kpep_db_create(nullptr, &db))) {
            EINSUMS_LOG_ERROR("Failed to load pmc db: {}", ret);
            return;
        }
        EINSUMS_LOG_INFO("Loaded pmc database: {} ({})", db->name, db->marketing_name);

        // create a config
        kpep_config *config{nullptr};
        if ((ret = kpep_config_create(db, &config))) {
            EINSUMS_LOG_ERROR("Failed to create kpep_config: {} ({})", ret, kpep_config_error_desc(ret));
            return;
        }
        if ((ret = kpep_config_force_counters(config))) {
            EINSUMS_LOG_ERROR("Failed to force counters: {} ({})", ret, kpep_config_error_desc(ret));
            return;
        }

        kpep_event *events[std::size(profile_events)];
        for (int i = 0; i < std::size(profile_events); i++) {
            EventAlias const *alias = profile_events + i;
            events[i]               = get_event(db, alias);
            if (!events[i]) {
                EINSUMS_LOG_ERROR("Cannot find event: {}", alias->alias);
                return;
            }
            EINSUMS_LOG_INFO("Found event: {} as {}", alias->alias, events[i]->name);
        }

        // add event to config
        for (auto event : events) {
            if ((ret = kpep_config_add_event(config, &event, 0, nullptr))) {
                EINSUMS_LOG_ERROR("Failed to add event: {}, {} ({})", event->name, ret, kpep_config_error_desc(ret));
                return;
            }
            EINSUMS_LOG_INFO("Added profile event: {}", event->name);
        }

        // prepare buffer and config
        u32   classes;
        usize register_count;
        if ((ret = kpep_config_kpc_classes(config, &classes))) {
            EINSUMS_LOG_ERROR("Failed to get kpc classes: {} ({})", ret, kpep_config_error_desc(ret));
            return;
        }
        if ((ret = kpep_config_kpc_count(config, &register_count))) {
            EINSUMS_LOG_ERROR("Failed to get kpc count: {} ({})", ret, kpep_config_error_desc(ret));
            return;
        }
        if ((ret = kpep_config_kpc_map(config, counter_map.data(), counter_map.size() * sizeof(usize)))) {
            EINSUMS_LOG_ERROR("Failed to get kpc map: {} ({})", ret, kpep_config_error_desc(ret));
            return;
        }
        if ((ret = kpep_config_kpc(config, registers.data(), registers.size() * sizeof(kpc_config_t)))) {
            EINSUMS_LOG_ERROR("Failed to get kpc registers: {} ({})", ret, kpep_config_error_desc(ret));
            return;
        }

        // set config to kernel
        if ((ret = kpc_force_all_ctrs_set(1))) {
            EINSUMS_LOG_ERROR("Failed force all ctrs: {}", ret);
            return;
        }
        if ((classes & KPC_CLASS_CONFIGURABLE_MASK) && register_count) {
            if ((ret = kpc_set_config(classes, registers.data()))) {
                EINSUMS_LOG_ERROR("Failed to set kpc config: {}", ret);
                return;
            }
        }

        // start counting
        if ((ret = kpc_set_counting(classes))) {
            EINSUMS_LOG_ERROR("Failed set counting: {}", ret);
            return;
        }
        if ((ret = kpc_set_thread_counting(classes))) {
            EINSUMS_LOG_ERROR("Failed set thread counting: {}", ret);
            return;
        }

        EINSUMS_LOG_INFO("Profile initialized and available");
        _available = true;
    }

    void start(std::vector<uint64_t> &s) override {
        if (!_available) [[unlikely]] {
            return;
        }
        if (kpc_get_thread_counters(0, KPC_MAX_COUNTERS, counters.data()) != 0) [[unlikely]] {
            EINSUMS_LOG_ERROR("Failed to get starting thread counters");
        }
        for (int i = 0; i < num_counters; ++i) {
            s[i] = counters[counter_map[i]];
        }
    }

    void stop(std::vector<uint64_t> &e) override {
        if (!_available) [[unlikely]] {
            return;
        }
        if (kpc_get_thread_counters(0, KPC_MAX_COUNTERS, counters.data()) != 0) [[unlikely]] {
            EINSUMS_LOG_ERROR("Failed to get stopping thread counters");
        }
        std::unordered_map<std::string, uint64_t> result;
        for (int i = 0; i < num_counters; ++i) {
            e[i] = counters[counter_map[i]];
        }
    }

    void delta(std::vector<uint64_t> const &s, std::vector<uint64_t> &e) const override {
        if (_available) [[likely]] {
            for (int i = 0; i < num_counters; ++i) {
                e[i] -= s[i];
            }
        }
    }

    int nevents() override { return num_counters; }

    std::vector<std::string> event_names() override {
        std::vector<std::string> names(num_counters);

        for (int i = 0; i < num_counters; ++i) {
            names[i] = profile_events[i].alias;
        }
        return names;
    }

  private:
    Library _library;
    bool    _available{false};
};

std::unique_ptr<PerformanceCounter> PerformanceCounter::create() {
    return std::make_unique<PerformanceCounterMac>();
}

} // namespace einsums::profile::detail