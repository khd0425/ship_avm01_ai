#include "recording_controller.hpp"

#include <cstdio>
#include <ctime>
#include <filesystem>

namespace avm::record {

std::string local_timestamp_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

RecordingController::RecordingController(RecordingConfig cfg, std::unique_ptr<IVideoSink> sink)
    : cfg_(std::move(cfg)), sink_(std::move(sink)) {}

bool RecordingController::enough_space() const {
    std::error_code ec;
    auto info = std::filesystem::space(cfg_.directory, ec);
    if (ec) return true;   // cannot tell: do not block recording
    return (info.available / (1024ull * 1024ull)) >= cfg_.min_free_mb;
}

bool RecordingController::start(int width, int height, const std::string& timestamp) {
    if (recording_) { error_ = "already recording"; return false; }
    if (!sink_) { error_ = "no video sink"; return false; }

    std::error_code ec;
    std::filesystem::create_directories(cfg_.directory, ec);
    if (ec) { error_ = "cannot create directory: " + cfg_.directory; return false; }
    if (!enough_space()) { error_ = "not enough free disk space"; return false; }

    path_ = (std::filesystem::path(cfg_.directory) / (cfg_.prefix + "_" + timestamp + cfg_.extension)).string();
    if (!sink_->open(path_, width, height, cfg_.fps)) {
        error_ = "cannot open video sink: " + path_;
        return false;
    }
    recording_ = true;
    frames_ = 0;
    frames_since_check_ = 0;
    error_.clear();
    return true;
}

void RecordingController::stop() {
    if (!recording_) return;
    sink_->close();
    recording_ = false;
}

bool RecordingController::on_frame(const void* data, std::size_t pitch_bytes, int channels) {
    if (!recording_) return false;
    // Re-check free space about once per 10 s of video instead of every frame.
    if (++frames_since_check_ >= static_cast<std::uint64_t>(cfg_.fps * 10.0)) {
        frames_since_check_ = 0;
        if (!enough_space()) {
            error_ = "disk full: recording stopped";
            stop();
            return false;
        }
    }
    if (!sink_->write(data, pitch_bytes, channels)) {
        error_ = "write failed: recording stopped";
        stop();
        return false;
    }
    ++frames_;
    return true;
}

} // namespace avm::record
