#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <memory>

#include "src/config/app_config.hpp"

namespace avm {

// ─── Image / Frame Types ────────────────────────────────────────────────────

enum class CameraType : uint8_t {
    EO_FISHEYE = 0,  // 180°+ ultra-wide electro-optical
    IR_THERMAL  = 1, // 45° long-wave infrared
};

enum class PixelFormat : uint8_t {
    YUV420 = 0,
    RGB8   = 1,
    RGBA8  = 2,
    MONO8  = 3,      // IR thermal typically 8-bit mono
    MONO16 = 4,      // Some IR sensors output 16-bit raw
};

struct CameraSpec {
    CameraType type;
    uint32_t   width;
    uint32_t   height;
    PixelFormat pixel_format;
    float      fps;
    uint8_t    gmsl_link_id;   // GMSL2 port index
};

struct CameraFrame {
    CameraSpec     spec;
    void*          stream{nullptr};  // cudaStream_t (kept opaque so this header has no CUDA dependency)

    // Mapped pinned memory (cudaHostAlloc(cudaHostAllocMapped)): on Jetson the
    // CPU and GPU share DRAM, so host_data and device_data alias the SAME
    // buffer (zero-copy, no cudaMemcpy). For plain device buffers host_data is null.
    void*          host_data{nullptr};
    size_t         host_pitch{0};

    // Device pointer used by CUDA kernels
    void*          device_data{nullptr};
    size_t         device_pitch{0};

    // Timing
    uint64_t       timestamp_us{0};
    uint32_t       frame_id{0};
};

// ─── IMU / NMEA Data ────────────────────────────────────────────────────────

// Vessel attitude (degrees). Conventions used by the stabilisation code:
//   pitch: bow-up positive     roll: starboard-side-down positive
struct EulerAngles {
    float yaw;
    float pitch;
    float roll;
};

struct IMUData {
    EulerAngles angles;
    float       gyro_x;   // rad/s
    float       gyro_y;
    float       gyro_z;
    uint64_t    timestamp_us;
};

struct GPSData {
    double latitude;
    double longitude;
    double speed_knots;
    double heading_true;
    uint64_t timestamp_us;
};

struct AISTarget {
    uint32_t    mmsi;
    double      latitude;
    double      longitude;
    float       sog;       // speed over ground (knots)
    float       cog;       // course over ground (degrees)
    float       heading;
    char        ship_name[32];
    uint8_t     ship_type;
};

// ─── Camera Parameters (pre-calibrated) ─────────────────────────────────────

struct Intrinsics {
    float fx, fy;          // focal length (pixels)
    float cx, cy;          // principal point
    float k1, k2, k3;     // radial distortion coefficients
    float p1, p2;         // tangential distortion coefficients
    float k4{0.0f};       // 4th fisheye coefficient (Kannala-Brandt, see geometry/view_math.hpp)
};

struct Extrinsics {
    // Rotation matrix (3×3) and translation vector
    std::array<float, 9> R;   // row-major
    std::array<float, 3> t;
};

struct CameraCalibration {
    CameraType  type;
    Intrinsics  intr;
    Extrinsics  extr;          // transform relative to vessel frame
    float       fov_degrees;
};

// ─── Remap / Alignment Data ─────────────────────────────────────────────────

struct RemapLUT {
    // Precomputed CUDA remap look-up table (2-channel float map)
    float* d_map_x{nullptr};
    float* d_map_y{nullptr};
    uint32_t width;
    uint32_t height;
};

struct BlendWeights {
    float* d_weights{nullptr}; // per-pixel alpha weight (0.0 ~ 1.0)
    uint32_t width;
    uint32_t height;
};

// ─── 3D Mesh / Viewport ─────────────────────────────────────────────────────

struct ViewportMesh {
    // Vertex buffer for the 3D surround-view mesh
    float* d_vertices{nullptr}; // [x, y, z] * N
    float* d_texcoords{nullptr};// [u, v] * N
    uint32_t* d_indices{nullptr};
    uint32_t num_vertices;
    uint32_t num_indices;
};

struct ViewTransform {
    std::array<float, 16> view_matrix;   // 4×4 row-major
    std::array<float, 16> proj_matrix;
    float fov_y;
    float aspect_ratio;
};

// ─── Detection / Inference ──────────────────────────────────────────────────

struct BoundingBox {
    float x, y, w, h;      // normalized [0, 1] relative to output size
    float confidence;
    int   class_id;
};

struct DetectionResult {
    std::vector<BoundingBox> boxes;
    uint64_t timestamp_us;
    uint32_t frame_id;
};

// ─── Pipeline Configuration ─────────────────────────────────────────────────
//
// Thin runtime view of config::AppConfig (the source of truth, loaded from
// config/avm_config.json). See make_pipeline_config() in avm_config.hpp.

struct PipelineConfig {
    config::AppConfig app;

    // EO camera (12 MP fisheye)
    CameraSpec eo_spec{
        CameraType::EO_FISHEYE, 4056, 3040,
        PixelFormat::RGB8, 30.0f, 0
    };

    // IR cameras (requirement spec: 2)
    std::vector<CameraSpec> ir_specs = {
        {CameraType::IR_THERMAL, 640, 480, PixelFormat::MONO8, 30.0f, 1},
        {CameraType::IR_THERMAL, 640, 480, PixelFormat::MONO8, 30.0f, 2},
    };

    // Output
    uint32_t output_width{1920};
    uint32_t output_height{1080};

    // Inference
    std::string model_path{"/etc/avm/models/maritime_detector.engine"};
    float detection_threshold{0.5f};

    // Features
    bool enable_ir_pip{true};        // FR-3.1 (mandatory)
    bool enable_ir_blending{false};  // FR-3.4 (optional, needs HW baseline <= 10 cm)
    bool enable_ar_overlay{true};    // FR-6 (optional)
    bool enable_inference{true};     // FR-7
};

} // namespace avm
