#pragma once

#include "src/config/app_config.hpp"
#include "src/core/error_checks.hpp"
#include "src/core/types.hpp"
#include "src/geometry/view_math.hpp"
#include "src/view/view_controller.hpp"

#include <atomic>
#include <mutex>

namespace avm {

// ─── MorphingRenderer ──────────────────────────────────────────────────────
//
// Renders the operator view straight from the raw EO fisheye frame with one
// GPU pass (FR-4, FR-5):
//   * three presets (forward perspective / top-view / ROI zoom) selectable
//     from the UI thread, blended smoothly when switching (view_controller),
//   * optional digital stabilisation: vessel roll/pitch from the IMU are fed
//     into the camera pose so the view stays level.
//
// Memory Flow:
//   Raw fisheye RGB8 (mapped pinned) ──> view kernel ──> RGBA8 output (mapped)
//   IMU + preset ──> ViewMapper (host, per frame, POD passed by value)
//
class MorphingRenderer {
public:
    MorphingRenderer(uint32_t output_width, uint32_t output_height, const config::AppConfig& app);

    MorphingRenderer(const MorphingRenderer&) = delete;
    MorphingRenderer& operator=(const MorphingRenderer&) = delete;

    /// Thread-safe: select preset 0..2 (applied at the next render()).
    void requestPreset(int index);
    /// Thread-safe: FR-5.1 on/off.
    void setStabilization(bool enabled);
    /// Thread-safe: latest vessel attitude.
    void updateAttitude(const IMUData& imu);

    /// Render the current view. `now_ms` is a monotonic clock in milliseconds.
    void render(const CameraFrame& raw_eo, CameraFrame& output, cudaStream_t stream, double now_ms);

    int activePreset() const { return active_preset_.load(); }
    bool stabilization() const { return stabilization_.load(); }
    /// True when the forward-view detections can be drawn 1:1 (preset 0, settled).
    bool detectionOverlayValid(double now_ms);
    int numPresets() const { return static_cast<int>(controller_.count()); }
    const char* presetName(int i) const { return controller_.name(i).c_str(); }

private:
    uint32_t out_w_, out_h_;
    geo::FisheyeModel model_;
    config::MountCfg mount_;

    std::mutex mu_;
    view::ViewController controller_;
    IMUData imu_{};

    std::atomic<int> requested_{0};
    std::atomic<int> active_preset_{0};
    std::atomic<bool> stabilization_{false};
    bool started_{false};
};

} // namespace avm
