#pragma once

// Runtime configuration (config/avm_config.json). Plain C++ (no CUDA/OpenCV) so
// it can be loaded, validated and unit-tested anywhere; the pipeline converts
// it into its own structures.

#include "src/geometry/view_math.hpp"
#include "src/imgproc/agc.hpp"
#include "src/record/recording_controller.hpp"
#include "src/sync/frame_sync.hpp"
#include "src/view/pip_layout.hpp"
#include "src/view/view_controller.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace avm::config {

struct CameraCfg {
    std::string type{"EO_FISHEYE"};
    std::uint32_t width{4056};   // 12.33 MP: matches the spec ch.3.3 bandwidth figures
    std::uint32_t height{3040};
    std::string pixel_format{"RGB8"};   // RGB8 | RGBA8 | MONO8 | MONO16 | YUV420
    double fps{30.0};
    int gmsl_link_id{0};
    /// Capture source: "/dev/videoN" or a full GStreamer pipeline string
    /// (e.g. "nvarguscamerasrc ! ... ! appsink"). Empty = simulated source.
    std::string source;
    std::string fourcc{"MJPG"};   // V4L2 pixel format request (USB cameras need MJPG for 1080p30)
    int exposure{-1};             // -1 = camera auto exposure; >= 0 = manual exposure (V4L2 units)
};

/// Simple pinhole model for the (normal-FOV) IR cameras.
struct PinholeCfg {
    double fx{400.0}, fy{400.0}, cx{320.0}, cy{240.0};
    double fov_deg{45.0};
};

struct MountCfg {
    double height_m{10.0};     // EO camera height above the water  (PLACEHOLDER: needs vessel data, spec ch.7 #6)
    double pitch_deg{20.0};    // tilt below the horizon           (PLACEHOLDER)
    double roll_deg{0.0};
};

struct OutputCfg {
    std::uint32_t width{1920};
    std::uint32_t height{1080};
};

struct PipCfg {
    bool enabled{true};
    std::vector<view::PipWindowCfg> windows;
};

struct ViewCfg {
    view::PresetSettings presets;
    bool stabilization{false};   // FR-5.1 (costs FPS: per-frame view mapping)
};

struct InferenceCfg {
    bool enabled{true};
    std::string arch{"yolox"};     // detector head: "yolox" (COCO pretrained) | "yolov8" (trained on the AVM classes)
    std::string source{"eo_view"}; // image the detector sees: "eo_view" (rectified fisheye view) | "ir0" | "ir1" (camera channel, e.g. a normal-FOV colour camera)
    std::string backend{"stub"};   // "stub" (no detection) | "opencv_dnn" (ONNX on CPU, dev PC) | "tensorrt" (Jetson, pending)
    std::string model_path{"/etc/avm/models/maritime_detector.engine"};
    int input_size{640};
    double confidence_threshold{0.5};
    double nms_threshold{0.45};
    int max_detections{100};
    std::string precision{"FP16"};
    double valid_range_m{50.0};              // FR-7.3
    std::vector<std::string> classes;
};

struct BlendCfg {                             // FR-3.4, optional, off by default
    bool enabled{false};
    double ir_alpha{0.35};
};

struct ArCfg {                                // FR-6, optional
    bool enabled{true};
    bool show_ais{true};
    bool show_detections{true};
    bool show_heading{true};
    bool show_collision_zone{true};
    double collision_zone_meters{500.0};
};

struct LoggingCfg {
    std::string directory{"logs"};
    std::string level{"info"};
    bool console{true};
};

struct AppConfig {
    CameraCfg eo;
    std::vector<CameraCfg> ir;
    OutputCfg output;
    geo::FisheyeModel eo_intrinsics;
    std::vector<PinholeCfg> ir_intrinsics;
    MountCfg mount;
    sync::SyncConfig sync;
    imgproc::AgcParams agc;
    PipCfg pip;
    ViewCfg view;
    InferenceCfg inference;
    BlendCfg blend;
    ArCfg ar;
    record::RecordingConfig recording;
    LoggingCfg logging;
};

/// Defaults derived from the requirement spec (1 EO 12 MP fisheye, 2 IR, 1080p out).
AppConfig default_app_config();

/// Parse configuration text. Missing keys keep their defaults. On failure
/// returns false and fills `err`.
bool parse_app_config(const std::string& json_text, AppConfig& cfg, std::string* err = nullptr);
bool load_app_config_file(const std::string& path, AppConfig& cfg, std::string* err = nullptr);

/// Read a fisheye calibration file written by tools/calib_fisheye.
bool load_fisheye_calibration_file(const std::string& path, geo::FisheyeModel& out,
                                   std::string* err = nullptr);
/// Serialise a calibration (the format calib_fisheye writes).
std::string fisheye_calibration_to_json(const geo::FisheyeModel& m, double rms_px, int num_images);

/// Human-readable problems (empty = valid).
std::vector<std::string> validate(const AppConfig& cfg);

} // namespace avm::config
