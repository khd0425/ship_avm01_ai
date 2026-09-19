#include "logger.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <vector>

namespace avm::util {

const char* to_string(LogLevel l) {
    switch (l) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

static std::string timestamp_ms() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));
    return buf;
}

static std::string date_stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    return buf;
}

bool Logger::open_directory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mu_);
    all_.close();
    errors_.close();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return false;

    const std::string day = date_stamp();
    all_path_   = (std::filesystem::path(dir) / ("avm_" + day + ".log")).string();
    error_path_ = (std::filesystem::path(dir) / ("avm_error_" + day + ".log")).string();
    all_.open(all_path_, std::ios::app);
    errors_.open(error_path_, std::ios::app);
    return all_.is_open() && errors_.is_open();
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(mu_);
    all_.close();
    errors_.close();
}

void Logger::log(LogLevel level, const char* component, const std::string& message) {
    if (level < level_) return;
    std::lock_guard<std::mutex> lock(mu_);
    std::string line = timestamp_ms() + " [" + to_string(level) + "] [" + component + "] " + message;
    if (level == LogLevel::Error) ++error_count_;
    if (console_) {
        (level >= LogLevel::Warn ? std::cerr : std::cout) << line << '\n';
    }
    if (all_.is_open()) { all_ << line << '\n'; all_.flush(); }
    if (level >= LogLevel::Warn && errors_.is_open()) { errors_ << line << '\n'; errors_.flush(); }
}

void Logger::logf(LogLevel level, const char* component, const char* fmt, ...) {
    if (level < level_) return;
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string msg;
    if (n > 0) {
        std::vector<char> buf(static_cast<std::size_t>(n) + 1);
        std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
        msg.assign(buf.data(), static_cast<std::size_t>(n));
    }
    va_end(ap2);
    log(level, component, msg);
}

std::string Logger::log_path() const {
    std::lock_guard<std::mutex> lock(mu_);
    return all_path_;
}

std::string Logger::error_log_path() const {
    std::lock_guard<std::mutex> lock(mu_);
    return error_path_;
}

} // namespace avm::util
