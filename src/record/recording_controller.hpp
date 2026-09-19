#pragma once

// Recording of the operator screen (FR-9.1): manual start/stop, file naming,
// free-space guard and error handling. The actual encoder is an IVideoSink;
// on Jetson the implementation is OpenCvVideoSink (GStreamer nvv4l2h264enc,
// hardware NVENC), tests use a fake sink.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace avm::record {

class IVideoSink {
public:
    virtual ~IVideoSink() = default;
    virtual bool open(const std::string& path, int width, int height, double fps) = 0;
    /// `bgr_or_rgba` = packed 8-bit image, `channels` = 3 or 4.
    virtual bool write(const void* data, std::size_t pitch_bytes, int channels) = 0;
    virtual void close() = 0;
};

struct RecordingConfig {
    std::string directory{"recordings"};
    std::string prefix{"avm"};
    std::string extension{".mp4"};
    std::uint64_t min_free_mb{512};   // refuse to start / stop when free space drops below
    double fps{30.0};
};

class RecordingController {
public:
    RecordingController(RecordingConfig cfg, std::unique_ptr<IVideoSink> sink);

    /// Start a new file "<dir>/<prefix>_YYYYMMDD_HHMMSS<ext>". `timestamp` is
    /// injected for testability ("20260919_120000"). False (and last_error()) on failure.
    bool start(int width, int height, const std::string& timestamp);
    void stop();
    bool is_recording() const { return recording_; }

    /// Feed one screen frame. Returns false if not recording or the write failed
    /// (a failed write stops the recording and sets last_error()).
    bool on_frame(const void* data, std::size_t pitch_bytes, int channels);

    const std::string& current_path() const { return path_; }
    const std::string& last_error() const { return error_; }
    std::uint64_t frames_written() const { return frames_; }

private:
    bool enough_space() const;

    RecordingConfig cfg_;
    std::unique_ptr<IVideoSink> sink_;
    bool recording_{false};
    std::string path_;
    std::string error_;
    std::uint64_t frames_{0};
    std::uint64_t frames_since_check_{0};
};

/// Current local time as "YYYYMMDD_HHMMSS".
std::string local_timestamp_string();

} // namespace avm::record
