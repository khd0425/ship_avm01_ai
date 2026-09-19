#pragma once

// Thread-safe file + console logger (FR-9.2: operation log and error log).
//
// open_directory() creates two files per day:
//   avm_YYYYMMDD.log         all messages at or above the configured level
//   avm_error_YYYYMMDD.log   WARN and ERROR messages only

#include <cstdarg>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <string>

namespace avm::util {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

const char* to_string(LogLevel l);

class Logger {
public:
    static Logger& instance();

    /// Open (append) the log files in `dir`. Creates the directory if needed.
    /// Returns false if the files cannot be opened; console logging still works.
    bool open_directory(const std::string& dir);
    void close();

    void set_level(LogLevel level) { level_ = level; }
    void set_console(bool enabled) { console_ = enabled; }

    void log(LogLevel level, const char* component, const std::string& message);
    void logf(LogLevel level, const char* component, const char* fmt, ...)
#if defined(__GNUC__)
        __attribute__((format(printf, 4, 5)))
#endif
        ;

    std::string log_path() const;
    std::string error_log_path() const;
    std::size_t error_count() const { return error_count_; }

private:
    Logger() = default;

    mutable std::mutex mu_;
    std::ofstream all_;
    std::ofstream errors_;
    std::string all_path_;
    std::string error_path_;
    LogLevel level_{LogLevel::Info};
    bool console_{true};
    std::size_t error_count_{0};
};

} // namespace avm::util

#define AVM_LOGD(comp, ...) ::avm::util::Logger::instance().logf(::avm::util::LogLevel::Debug, comp, __VA_ARGS__)
#define AVM_LOGI(comp, ...) ::avm::util::Logger::instance().logf(::avm::util::LogLevel::Info,  comp, __VA_ARGS__)
#define AVM_LOGW(comp, ...) ::avm::util::Logger::instance().logf(::avm::util::LogLevel::Warn,  comp, __VA_ARGS__)
#define AVM_LOGE(comp, ...) ::avm::util::Logger::instance().logf(::avm::util::LogLevel::Error, comp, __VA_ARGS__)
