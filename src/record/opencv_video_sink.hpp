#pragma once

// Video encoder for RecordingController on top of cv::VideoWriter.
// On Jetson the H.264 encode runs on the hardware NVENC block through a
// GStreamer pipeline (nvv4l2h264enc); on other machines it falls back to
// OpenCV's software mp4v writer. Built only with AVM_WITH_OPENCV.

#ifdef AVM_WITH_OPENCV

#include "src/record/recording_controller.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace avm::record {

class OpenCvVideoSink final : public IVideoSink {
public:
    explicit OpenCvVideoSink(int bitrate_kbps = 8000) : bitrate_kbps_(bitrate_kbps) {}

    bool open(const std::string& path, int width, int height, double fps) override;
    bool write(const void* data, std::size_t pitch_bytes, int channels) override;
    void close() override;

private:
    cv::VideoWriter writer_;
    cv::Mat bgr_;
    int width_{0}, height_{0};
    int bitrate_kbps_;
};

} // namespace avm::record

#endif
