#include "morphing_renderer.hpp"

#include "src/render/cuda_view_render.hpp"
#include "src/util/logger.hpp"

namespace avm {

namespace {

// Presets fitted to what the camera really sees (no black areas), see view::make_default_presets.
std::vector<view::ViewPreset> build_presets(const config::AppConfig& app, uint32_t w, uint32_t h) {
    view::FitContext fit;
    fit.model = app.eo_intrinsics;
    fit.pose = geo::make_camera_pose(app.mount.height_m, app.mount.pitch_deg, app.mount.roll_deg);
    fit.src_w = app.eo_intrinsics.width > 0 ? app.eo_intrinsics.width : static_cast<int>(app.eo.width);
    fit.src_h = app.eo_intrinsics.height > 0 ? app.eo_intrinsics.height : static_cast<int>(app.eo.height);
    std::string report;
    auto presets = view::make_default_presets(app.view.presets, app.mount.height_m, static_cast<int>(w),
                                              static_cast<int>(h), &fit, &report);
    if (!report.empty()) AVM_LOGI("render", "views fitted to the visible area: %s", report.c_str());
    return presets;
}

} // namespace

MorphingRenderer::MorphingRenderer(uint32_t output_width, uint32_t output_height,
                                   const config::AppConfig& app)
    : out_w_(output_width)
    , out_h_(output_height)
    , model_(app.eo_intrinsics)
    , mount_(app.mount)
    , controller_(build_presets(app, output_width, output_height), app.view.presets.transition_ms) {
    stabilization_ = app.view.stabilization;
    if (app.view.presets.default_preset >= 0 &&
        app.view.presets.default_preset < static_cast<int>(controller_.count()))
        requested_ = app.view.presets.default_preset;
}

void MorphingRenderer::requestPreset(int index) {
    if (index >= 0 && index < numPresets()) requested_ = index;
}

void MorphingRenderer::setStabilization(bool enabled) { stabilization_ = enabled; }

void MorphingRenderer::updateAttitude(const IMUData& imu) {
    std::lock_guard<std::mutex> lk(mu_);
    imu_ = imu;
}

bool MorphingRenderer::detectionOverlayValid(double now_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    return started_ && controller_.target() == 0 && !controller_.in_transition(now_ms);
}

void MorphingRenderer::render(const CameraFrame& raw_eo, CameraFrame& output,
                              cudaStream_t stream, double now_ms) {
    geo::ViewMapper<float> mapper;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const int want = requested_.load();
        if (!started_ || want != controller_.target()) {
            controller_.select(want, now_ms);   // first call: instant, later: smooth fly-through
            started_ = true;
        }
        active_preset_ = controller_.target();

        const bool stab = stabilization_.load();
        const geo::CameraPose pose = geo::make_camera_pose(
            mount_.height_m, mount_.pitch_deg, mount_.roll_deg,
            stab ? imu_.angles.pitch : 0.0, stab ? imu_.angles.roll : 0.0);
        mapper = geo::make_view_mapper<float>(model_, pose, controller_.current(now_ms),
                                              static_cast<int>(out_w_), static_cast<int>(out_h_));
    }

    const int err = render::launch_render_view_rgb8_to_rgba(
        static_cast<const uint8_t*>(raw_eo.device_data), static_cast<int>(raw_eo.device_pitch),
        static_cast<int>(raw_eo.spec.width), static_cast<int>(raw_eo.spec.height),
        static_cast<uint8_t*>(output.device_data), static_cast<int>(output.device_pitch),
        static_cast<int>(out_w_), static_cast<int>(out_h_), mapper, stream);
    if (err != 0) {
        AVM_LOGE("render", "view kernel launch failed (cudaError %d)", err);
    }
}

} // namespace avm
