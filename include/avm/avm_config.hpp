#pragma once

#include "src/config/app_config.hpp"
#include "src/core/types.hpp"
#include <cstdlib>
#include <string>

namespace avm {

// ─── Compile-time tunable constants ─────────────────────────────────────────

namespace constants {

// Pipeline timing targets (PR-1 / PR-2)
inline constexpr float    TARGET_FPS             = 30.0f;
inline constexpr float    FRAME_BUDGET_MS        = 33.0f;  // ≈ 1000/30

// CUDA stream allocation
inline constexpr int      NUM_CUDA_STREAMS       = 4;
enum StreamIndex : int {
    STREAM_CAPTURE    = 0,
    STREAM_UNDISTORT  = 1,
    STREAM_ALIGN      = 2,
    STREAM_INFERENCE  = 3,
};

// Undistortion
inline constexpr int      REMAP_LUT_INTERP       = 1;  // 0=nearest, 1=linear

// Inference
inline constexpr int      DETECTION_INPUT_SIZE   = 640;
inline constexpr int      MAX_DETECTIONS         = 100;
inline constexpr float    NMS_THRESHOLD          = 0.45f;

// AR overlay
inline constexpr float    HUD_LINE_WIDTH         = 2.0f;

} // namespace constants

// ─── Runtime configuration ──────────────────────────────────────────────────

inline PixelFormat parse_pixel_format(const std::string& s) {
    if (s == "RGBA8") return PixelFormat::RGBA8;
    if (s == "MONO8") return PixelFormat::MONO8;
    if (s == "MONO16") return PixelFormat::MONO16;
    if (s == "YUV420") return PixelFormat::YUV420;
    return PixelFormat::RGB8;
}

inline CameraSpec to_camera_spec(const config::CameraCfg& c) {
    CameraType t = (c.type == "IR_THERMAL") ? CameraType::IR_THERMAL : CameraType::EO_FISHEYE;
    return CameraSpec{t, c.width, c.height, parse_pixel_format(c.pixel_format),
                      static_cast<float>(c.fps), static_cast<uint8_t>(c.gmsl_link_id)};
}

/// Environment overrides (deployment convenience); applied on top of the file.
inline void apply_env_overrides(config::AppConfig& a) {
    if (auto* v = std::getenv("AVM_OUTPUT_WIDTH"))  a.output.width  = static_cast<uint32_t>(std::atoi(v));
    if (auto* v = std::getenv("AVM_OUTPUT_HEIGHT")) a.output.height = static_cast<uint32_t>(std::atoi(v));
    if (auto* v = std::getenv("AVM_MODEL_PATH"))    a.inference.model_path = v;
    if (auto* v = std::getenv("AVM_DETECTION_THRESHOLD")) a.inference.confidence_threshold = std::atof(v);
    if (auto* v = std::getenv("AVM_DISABLE_IR"))    { if (std::atoi(v) != 0) a.pip.enabled = false; }
    if (auto* v = std::getenv("AVM_DISABLE_AR"))    { if (std::atoi(v) != 0) a.ar.enabled = false; }
    if (auto* v = std::getenv("AVM_DISABLE_INFERENCE")) { if (std::atoi(v) != 0) a.inference.enabled = false; }
}

inline PipelineConfig make_pipeline_config(const config::AppConfig& app) {
    PipelineConfig cfg;
    cfg.app = app;
    cfg.eo_spec = to_camera_spec(app.eo);
    cfg.ir_specs.clear();
    for (const auto& ir : app.ir) cfg.ir_specs.push_back(to_camera_spec(ir));
    cfg.output_width = app.output.width;
    cfg.output_height = app.output.height;
    cfg.model_path = app.inference.model_path;
    cfg.detection_threshold = static_cast<float>(app.inference.confidence_threshold);
    cfg.enable_ir_pip = app.pip.enabled;
    cfg.enable_ir_blending = app.blend.enabled;
    cfg.enable_ar_overlay = app.ar.enabled;
    cfg.enable_inference = app.inference.enabled;
    return cfg;
}

inline PipelineConfig load_default_config() {
    config::AppConfig app = config::default_app_config();
    apply_env_overrides(app);
    return make_pipeline_config(app);
}

} // namespace avm
