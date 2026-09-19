#pragma once

#include "src/core/types.hpp"
#include "src/core/error_checks.hpp"
#include <memory>
#include <vector>

namespace avm {

// ─── AROverlay ─────────────────────────────────────────────────────────────
//
// Renders real-time AR HUD overlay on the AVM output:
//   - AIS target markers (position, heading, speed, MMSI label)
//   - Vessel heading indicator
//   - Dynamic collision guidelines (CPA / TCPA zones)
//   - Distance tags
//   - Detection bounding boxes from InferenceEngine
//
// Memory Flow:
//   Final image (device) ← CUDA overlay kernel ← HUD primitives (device)
//
class AROverlay {
public:
    explicit AROverlay(uint32_t output_width,
                       uint32_t output_height,
                       const CameraCalibration& eo_calib);
    ~AROverlay();

    AROverlay(const AROverlay&) = delete;
    AROverlay& operator=(const AROverlay&) = delete;

    /// Update AIS targets for rendering.
    void updateTargets(const std::vector<AISTarget>& targets);

    /// Update detection results.
    void updateDetections(const DetectionResult& detections);

    /// Update vessel navigation data.
    void updateNavigation(const GPSData& gps, float heading_deg);

    /// Render all AR elements onto the output image.
    /// @param image  Output image buffer (device, RGBA8 or RGB8)
    /// @param stream CUDA stream
    void render(CameraFrame& image, cudaStream_t stream);

    /// Enable/disable individual overlay elements.
    void setShowAIS(bool show)       { show_ais_ = show; }
    void setShowDetections(bool show){ show_detections_ = show; }
    void setShowHeading(bool show)   { show_heading_ = show; }
    void setShowCollision(bool show) { show_collision_ = show; }

private:
    struct AISRenderData {
        float x_norm, y_norm;       // normalized screen position
        float heading_rad;
        float sog_knots;
        uint32_t mmsi;
        char label[24];
        float cpa_nm;               // closest point of approach (NM)
        float tcpa_min;             // time to CPA (minutes)
    };

    uint32_t out_w_, out_h_;
    CameraCalibration eo_calib_;

    // GPU buffers
    float* d_ais_vertices_{nullptr};
    uint32_t* d_ais_indices_{nullptr};
    int num_ais_{0};

    std::vector<AISRenderData> ais_render_data_;

    // State
    GPSData current_gps_{};
    float current_heading_{0.0f};
    DetectionResult current_detections_;

    // Toggles
    bool show_ais_{true};
    bool show_detections_{true};
    bool show_heading_{true};
    bool show_collision_{true};
};

} // namespace avm
