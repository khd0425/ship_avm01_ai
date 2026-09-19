#pragma once

#include "src/core/types.hpp"
#include "src/core/error_checks.hpp"
#include "src/geometry/lut_builder.hpp"
#include "src/geometry/view_math.hpp"
#include <memory>

namespace avm {

// ─── UndistortEngine ────────────────────────────────────────────────────────
//
// LUT based fisheye distortion correction (FR-2.3). The LUT is either built
// from the calibration (buildLUTs) or loaded from a file produced by
// tools/build_lut (uploadLUT). It is used for the rectified forward view that
// feeds the AI detector; the operator views (presets, top-view, stabilisation)
// are rendered directly from the raw fisheye by MorphingRenderer.
//
// Memory Flow:
//   Raw fisheye (mapped pinned, zero-copy) ──> remap kernel (LUT) ──> rectified buffer (device)
//
class UndistortEngine {
public:
    UndistortEngine(const geo::FisheyeModel& model, const geo::CameraPose& pose,
                    const geo::ViewParams& view,
                    uint32_t output_width, uint32_t output_height,
                    uint32_t src_width, uint32_t src_height);
    ~UndistortEngine() = default;

    UndistortEngine(const UndistortEngine&) = delete;
    UndistortEngine& operator=(const UndistortEngine&) = delete;
    UndistortEngine(UndistortEngine&&) = delete;
    UndistortEngine& operator=(UndistortEngine&&) = delete;

    /// Build the LUT on the host from the calibration and upload it.
    bool buildLUTs(cudaStream_t stream);

    /// Upload a prebuilt LUT (e.g. from an .avmlut file). Size must match.
    bool uploadLUT(const geo::LutMap& lut, cudaStream_t stream);

    /// Execute distortion correction (async on `stream`).
    void correct(const CameraFrame& input, CameraFrame& output, cudaStream_t stream);

    bool ready() const { return luts_built_; }

private:
    geo::FisheyeModel model_;
    geo::CameraPose pose_;
    geo::ViewParams view_;
    uint32_t out_width_, out_height_, src_width_, src_height_;

    unique_cuda_ptr d_map_x_;
    unique_cuda_ptr d_map_y_;
    bool luts_built_{false};
};

} // namespace avm
