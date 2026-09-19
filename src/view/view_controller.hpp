#pragma once

// Viewpoint presets and smooth transitions (FR-4.1 .. FR-4.5).
//
//   preset 0  forward perspective  (fisheye-corrected forward view)
//   preset 1  near-range top-view  (water plane, radius 15 m by default)
//   preset 2  region-of-interest zoom
//   preset 3  raw fisheye (original image as captured, no geometry; diagnostic / as-is view)
//
// A view is a virtual pinhole camera (see geometry/view_math.hpp), so a
// transition is a fly-through: position, orientation and field of view are
// eased between the two presets. Re-selecting during a transition starts
// from the *currently displayed* view, so the picture never jumps.

#include "src/geometry/view_math.hpp"

#include <string>
#include <vector>

namespace avm::view {

struct ViewPreset {
    std::string name;
    geo::ViewParams view;
};

/// Parameters the default presets are derived from (mirrors config "view").
struct PresetSettings {
    double forward_hfov_deg{110.0};
    double forward_pitch_deg{10.0};
    double topview_radius_m{15.0};        // FR-4.3
    double topview_vfov_deg{60.0};
    double topview_center_forward_m{0.0}; // shift of the BEV centre from the EO camera
    double roi_yaw_deg{0.0};
    double roi_pitch_deg{15.0};
    double roi_hfov_deg{40.0};
    double transition_ms{700.0};
    bool fit_to_visible{true};            // no black areas: fit every view to what the camera really sees
    bool raw_fill{true};                  // raw view: crop to the image circle (false = whole sensor image)
    int default_preset{0};                // preset shown at start (3 = raw fisheye as captured)
};

/// What a preset needs to be fitted to the camera's real field of view.
struct FitContext {
    geo::FisheyeModel model;
    geo::CameraPose pose;      // static mount pose (no IMU)
    int src_w{0}, src_h{0};    // captured image size
};

/// Fraction (0..1) of the output pixels of `view` that show real camera data
/// (inside the lens FOV and the captured image). Sampled on a coarse grid.
double visible_fraction(const FitContext& fit, const geo::ViewParams& view, int out_w, int out_h,
                        int grid_w = 96, int grid_h = 54);

/// Build the presets (forward, top-view, ROI zoom, raw fisheye) for an output of out_w x out_h
/// pixels and an EO camera mounted `camera_height_m` above the water. With a FitContext (and
/// s.fit_to_visible) each preset is adjusted so that no output pixel is black: the top-view
/// centre moves forward (or its radius shrinks), perspective views narrow their field of view.
/// `report` receives a human-readable description of what was changed.
std::vector<ViewPreset> make_default_presets(const PresetSettings& s, double camera_height_m,
                                             int out_w, int out_h, const FitContext* fit = nullptr,
                                             std::string* report = nullptr);

double smoothstep(double t);
geo::ViewParams lerp_view(const geo::ViewParams& a, const geo::ViewParams& b, double t);

class ViewController {
public:
    ViewController(std::vector<ViewPreset> presets, double transition_ms = 700.0);

    /// Switch to preset `index` at time `now_ms`. False if index is invalid.
    bool select(int index, double now_ms);

    /// The view to display at `now_ms` (interpolated during a transition).
    geo::ViewParams current(double now_ms) const;

    bool in_transition(double now_ms) const;
    int target() const { return target_; }
    std::size_t count() const { return presets_.size(); }
    const std::string& name(int index) const { return presets_.at(static_cast<std::size_t>(index)).name; }

private:
    std::vector<ViewPreset> presets_;
    double transition_ms_;
    int target_{0};
    geo::ViewParams from_{};
    geo::ViewParams to_{};
    double t0_ms_{0.0};
    bool started_{false};
};

} // namespace avm::view
