#ifdef AVM_WITH_OPENCV

#include "opencv_video_sink.hpp"

#include "src/util/logger.hpp"

#include <opencv2/imgproc.hpp>

namespace avm::record {

bool OpenCvVideoSink::open(const std::string& path, int width, int height, double fps) {
    width_ = width;
    height_ = height;
    const cv::Size size(width, height);

    // 1) Jetson hardware encoder (NVENC) through GStreamer.
    const std::string gst =
        "appsrc ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate=" +
        std::to_string(bitrate_kbps_ * 1000) + " ! h264parse ! qtmux ! filesink location=" + path;
    if (writer_.open(gst, cv::CAP_GSTREAMER, 0, fps, size, true)) {
        AVM_LOGI("record", "recording with NVENC: %s", path.c_str());
        return true;
    }
    // 2) Software fallback (development machines).
    if (writer_.open(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size, true)) {
        AVM_LOGW("record", "NVENC pipeline unavailable, using software mp4v: %s", path.c_str());
        return true;
    }
    return false;
}

bool OpenCvVideoSink::write(const void* data, std::size_t pitch_bytes, int channels) {
    if (!writer_.isOpened()) return false;
    cv::Mat src(height_, width_, channels == 4 ? CV_8UC4 : CV_8UC3, const_cast<void*>(data), pitch_bytes);
    if (channels == 4) {
        cv::cvtColor(src, bgr_, cv::COLOR_RGBA2BGR);
        writer_.write(bgr_);
    } else {
        cv::cvtColor(src, bgr_, cv::COLOR_RGB2BGR);
        writer_.write(bgr_);
    }
    return true;
}

void OpenCvVideoSink::close() { writer_.release(); }

} // namespace avm::record

#endif
