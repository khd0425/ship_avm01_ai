#pragma once

#include "src/core/types.hpp"
#include "src/core/error_checks.hpp"
#include <memory>
#include <vector>

namespace avm {

// ─── AlignmentEngine ────────────────────────────────────────────────────────
//
// Projects 45° IR thermal streams onto the 180° EO fisheye frame using
// homography + extrinsic matrix transformations, then performs dynamic
// alpha blending via custom CUDA kernels.
//
// Memory Flow:
//   Undistorted EO frame (device) ───┐
//   Undistorted IR frame (device) ───┤
//                                    ├── CUDA Blend Kernel → Blended output
//   Precomputed homography H (dev) ──┘
//   Precomputed weight map (dev) ────┘
//
class AlignmentEngine {
public:
    explicit AlignmentEngine(
        const CameraCalibration& eo_calib,
        const std::vector<CameraCalibration>& ir_calibs,
        uint32_t output_width,
        uint32_t output_height);
    ~AlignmentEngine();

    AlignmentEngine(const AlignmentEngine&) = delete;
    AlignmentEngine& operator=(const AlignmentEngine&) = delete;
    AlignmentEngine(AlignmentEngine&&) = default;
    AlignmentEngine& operator=(AlignmentEngine&&) = default;

    /// Precompute homography matrices and weight maps for each IR camera.
    /// Must be called once before processing.
    bool buildMaps(cudaStream_t stream);

    /// Blend one IR frame onto the EO frame.
    /// @param eo_frame  Undistorted EO image (device)
    /// @param ir_frame  Undistorted IR image (device)
    /// @param ir_index  Which IR camera (0..N-1)
    /// @param output    Blended output (device, pre-allocated)
    /// @param stream    CUDA stream
    void blend(const CameraFrame& eo_frame,
               const CameraFrame& ir_frame,
               int ir_index,
               CameraFrame& output,
               cudaStream_t stream);

    /// Blend all IR frames onto the EO frame in one call.
    void blendAll(const CameraFrame& eo_frame,
                  const std::vector<CameraFrame>& ir_frames,
                  CameraFrame& output,
                  cudaStream_t stream);

private:
    CameraCalibration eo_calib_;
    std::vector<CameraCalibration> ir_calibs_;
    uint32_t out_w_, out_h_;

    // Per-IR-camera precomputed data
    struct IRMap {
        float* d_homography{nullptr}; // 3×3 matrix (row-major)
        float* d_weight_map{nullptr}; // per-pixel blend weight
        uint32_t src_w, src_h;
        bool valid{false};
    };
    std::vector<IRMap> ir_maps_;

    // Shared weight map for the EO frame itself
    BlendWeights eo_weights_{};

    bool maps_built_{false};
};

} // namespace avm
