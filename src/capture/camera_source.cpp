#include "camera_source.hpp"

#include "src/util/clock.hpp"
#include "src/util/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef AVM_WITH_OPENCV
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace avm::capture {

namespace {

// ─── Synthetic source ───────────────────────────────────────────────────────
class SimulatedSource final : public ICameraSource {
public:
    SimulatedSource(const config::CameraCfg& cfg, bool is_eo, int index)
        : cfg_(cfg), is_eo_(is_eo), name_((is_eo ? "sim-EO" : "sim-IR") + std::to_string(index)) {}

    bool open() override {
        w_ = static_cast<int>(cfg_.width);
        h_ = static_cast<int>(cfg_.height);
        bpp_ = cfg_.pixel_format == "MONO16" ? 2 : cfg_.pixel_format == "MONO8" ? 1
             : cfg_.pixel_format == "RGBA8" ? 4 : 3;
        row_bytes_ = static_cast<std::size_t>(w_) * bpp_;
        build_base();
        period_us_ = static_cast<std::uint64_t>(1e6 / std::max(1.0, cfg_.fps));
        // Align to a common grid so simulated channels are in sync like hardware-triggered ones.
        const std::uint64_t now = util::steady_now_us();
        next_us_ = (now / period_us_ + 1) * period_us_;
        frame_ = 0;
        return true;
    }

    bool grab(std::uint8_t* dst, std::size_t pitch, std::uint64_t& ts) override {
        std::this_thread::sleep_for(std::chrono::microseconds(
            next_us_ > util::steady_now_us() ? next_us_ - util::steady_now_us() : 0));
        ts = next_us_;
        next_us_ += period_us_;
        ++frame_;

        for (int y = 0; y < h_; ++y)
            std::memcpy(dst + static_cast<std::size_t>(y) * pitch,
                        base_.data() + static_cast<std::size_t>(y) * row_bytes_, row_bytes_);

        // moving vertical bar so a live picture is visibly alive
        const int bar_w = std::max(4, w_ / 64);
        const int x0 = static_cast<int>((frame_ * 12) % static_cast<std::uint64_t>(std::max(1, w_ - bar_w)));
        for (int y = 0; y < h_; ++y) {
            std::uint8_t* row = dst + static_cast<std::size_t>(y) * pitch + static_cast<std::size_t>(x0) * bpp_;
            std::memset(row, 255, static_cast<std::size_t>(bar_w) * bpp_);
        }
        return true;
    }

    void close() override {}
    const char* name() const override { return name_.c_str(); }

private:
    void build_base() {
        base_.assign(row_bytes_ * static_cast<std::size_t>(h_), 0);
        const double cx = 0.5 * (w_ - 1), cy = 0.5 * (h_ - 1);
        const double r_circle = 0.5 * std::min(w_, h_);
        for (int y = 0; y < h_; ++y) {
            std::uint8_t* row = base_.data() + static_cast<std::size_t>(y) * row_bytes_;
            for (int x = 0; x < w_; ++x) {
                const double r = std::hypot(x - cx, y - cy);
                if (is_eo_) {
                    // image circle with rings every 15 deg-equivalent and a horizon-like gradient
                    const bool inside = r <= r_circle;
                    const bool ring = inside && std::fmod(r, r_circle / 6.0) < 3.0;
                    std::uint8_t px[4] = {
                        static_cast<std::uint8_t>(inside ? 40 + 120 * x / w_ : 0),
                        static_cast<std::uint8_t>(inside ? 60 + 100 * y / h_ : 0),
                        static_cast<std::uint8_t>(inside ? 120 : 0), 255};
                    if (ring) px[0] = px[1] = px[2] = 230;
                    std::memcpy(row + static_cast<std::size_t>(x) * bpp_, px, static_cast<std::size_t>(std::min(bpp_, 4)));
                } else {
                    const double blob = std::exp(-(r * r) / (2.0 * (0.15 * w_) * (0.15 * w_)));
                    const double v = 0.25 + 0.55 * blob + 0.1 * x / w_;
                    if (bpp_ == 2) {
                        const std::uint16_t v16 = static_cast<std::uint16_t>(28000 + v * 8000.0);
                        std::memcpy(row + static_cast<std::size_t>(x) * 2, &v16, 2);
                    } else {
                        row[static_cast<std::size_t>(x) * bpp_] = static_cast<std::uint8_t>(90 + v * 60.0);
                    }
                }
            }
        }
    }

    config::CameraCfg cfg_;
    bool is_eo_;
    std::string name_;
    int w_{0}, h_{0}, bpp_{3};
    std::size_t row_bytes_{0};
    std::vector<std::uint8_t> base_;
    std::uint64_t period_us_{33333}, next_us_{0}, frame_{0};
};

#ifdef AVM_WITH_OPENCV
// ─── OpenCV / GStreamer source (Jetson production path) ─────────────────────
class OpenCvSource final : public ICameraSource {
public:
    OpenCvSource(const config::CameraCfg& cfg, bool is_eo, int index)
        : cfg_(cfg), is_eo_(is_eo), name_(cfg.source) { (void)index; }

    bool open() override {
        bool ok = false;
        if (cfg_.source.rfind("/dev/video", 0) == 0) {
            ok = cap_.open(cfg_.source, cv::CAP_V4L2);
            if (ok) {
                if (cfg_.fourcc.size() == 4)
                    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc(cfg_.fourcc[0], cfg_.fourcc[1],
                                                                          cfg_.fourcc[2], cfg_.fourcc[3]));
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, cfg_.width);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.height);
                cap_.set(cv::CAP_PROP_FPS, cfg_.fps);
                if (cfg_.pixel_format == "MONO16") cap_.set(cv::CAP_PROP_CONVERT_RGB, 0);
                if (cfg_.exposure >= 0) {                     // V4L2: 1 = manual, 3 = aperture-priority auto
                    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
                    cap_.set(cv::CAP_PROP_EXPOSURE, cfg_.exposure);
                } else {
                    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);
                }
            }
        } else {
            // Full GStreamer pipeline ending in appsink, e.g.
            //  "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=4056,height=3040,framerate=30/1 !
            //   nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
            ok = cap_.open(cfg_.source, cv::CAP_GSTREAMER);
        }
        if (!ok) AVM_LOGW("capture", "cannot open camera source '%s'", cfg_.source.c_str());
        return ok;
    }

    bool grab(std::uint8_t* dst, std::size_t pitch, std::uint64_t& ts) override {
        if (!cap_.isOpened() || !cap_.read(frame_) || frame_.empty()) return false;
        ts = util::steady_now_us();   // OpenCV exposes no sensor timestamp; capture-thread time is used

        const int w = static_cast<int>(cfg_.width), h = static_cast<int>(cfg_.height);
        const cv::Size size(w, h);
        if (cfg_.pixel_format == "MONO16") {
            if (frame_.type() != CV_16UC1) return false;
            cv::Mat out(h, w, CV_16UC1, dst, pitch);
            if (frame_.size() == size) frame_.copyTo(out); else { cv::resize(frame_, tmp_, size); tmp_.copyTo(out); }
        } else if (cfg_.pixel_format == "MONO8") {
            cv::Mat out(h, w, CV_8UC1, dst, pitch);
            const cv::Mat* src = &frame_;
            if (frame_.size() != size) { cv::resize(frame_, tmp_, size); src = &tmp_; }
            if (src->channels() == 3) cv::cvtColor(*src, out, cv::COLOR_BGR2GRAY);
            else if (src->channels() == 1) src->copyTo(out);
            else return false;
        } else {   // RGB8 (EO): OpenCV delivers BGR
            if (frame_.type() != CV_8UC3) return false;
            cv::Mat out(h, w, CV_8UC3, dst, pitch);
            const cv::Mat* src = &frame_;
            if (frame_.size() != size) { cv::resize(frame_, tmp_, size); src = &tmp_; }
            cv::cvtColor(*src, out, cv::COLOR_BGR2RGB);
        }
        return true;
    }

    void close() override { cap_.release(); }
    const char* name() const override { return name_.c_str(); }

private:
    config::CameraCfg cfg_;
    bool is_eo_;
    std::string name_;
    cv::VideoCapture cap_;
    cv::Mat frame_, tmp_;
};
#endif

} // namespace

std::unique_ptr<ICameraSource> make_camera_source(const config::CameraCfg& cfg, bool is_eo, int index) {
    if (cfg.source.empty()) return std::make_unique<SimulatedSource>(cfg, is_eo, index);
#ifdef AVM_WITH_OPENCV
    return std::make_unique<OpenCvSource>(cfg, is_eo, index);
#else
    AVM_LOGW("capture", "camera source '%s' needs OpenCV (AVM_WITH_OPENCV); using simulated source",
             cfg.source.c_str());
    return std::make_unique<SimulatedSource>(cfg, is_eo, index);
#endif
}

} // namespace avm::capture
