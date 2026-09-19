#include "view_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace avm::view {

double smoothstep(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

geo::ViewParams lerp_view(const geo::ViewParams& a, const geo::ViewParams& b, double t) {
    auto mix = [t](double x, double y) { return x + (y - x) * t; };
    geo::ViewParams r;
    r.kind = (t < 0.5) ? a.kind : b.kind;   // projection kind cannot be blended
    r.yaw_deg = mix(a.yaw_deg, b.yaw_deg);
    r.pitch_deg = mix(a.pitch_deg, b.pitch_deg);
    r.roll_deg = mix(a.roll_deg, b.roll_deg);
    r.hfov_deg = mix(a.hfov_deg, b.hfov_deg);
    r.offset_forward_m = mix(a.offset_forward_m, b.offset_forward_m);
    r.offset_right_m = mix(a.offset_right_m, b.offset_right_m);
    r.offset_up_m = mix(a.offset_up_m, b.offset_up_m);
    return r;
}

double visible_fraction(const FitContext& fit, const geo::ViewParams& view, int out_w, int out_h,
                        int grid_w, int grid_h) {
    const auto mapper = geo::make_view_mapper<double>(fit.model, fit.pose, view, out_w, out_h);
    int ok = 0, total = 0;
    for (int j = 0; j < grid_h; ++j) {
        const double v = (j + 0.5) * out_h / grid_h;
        for (int i = 0; i < grid_w; ++i) {
            const double u = (i + 0.5) * out_w / grid_w;
            double su, sv;
            ++total;
            if (mapper.map(u, v, su, sv) && su >= 0.0 && sv >= 0.0 && su <= fit.src_w - 1 && sv <= fit.src_h - 1)
                ++ok;
        }
    }
    return total ? static_cast<double>(ok) / total : 0.0;
}

namespace {

constexpr double kFullyVisible = 0.999;

// Top-view centred `c` metres ahead of the EO camera with a vertical half-extent of `radius` metres.
geo::ViewParams make_topview(double radius, double vfov_deg, double c, double camera_h, int out_w, int out_h) {
    const double half_v = 0.5 * vfov_deg * geo::kDegToRad;
    const double h_virtual = radius / std::tan(half_v);
    const double half_h = std::atan(std::tan(half_v) * static_cast<double>(out_w) / std::max(1, out_h));
    geo::ViewParams v;
    v.pitch_deg = 90.0;
    v.hfov_deg = 2.0 * half_h / geo::kDegToRad;
    v.offset_up_m = h_virtual - camera_h;
    v.offset_forward_m = c;
    return v;
}

} // namespace

std::vector<ViewPreset> make_default_presets(const PresetSettings& s, double camera_height_m,
                                             int out_w, int out_h, const FitContext* fit,
                                             std::string* report) {
    std::vector<ViewPreset> p(4);
    std::string note;
    const bool do_fit = fit && s.fit_to_visible;

    p[0].name = "forward";
    p[0].view.hfov_deg = s.forward_hfov_deg;
    p[0].view.pitch_deg = s.forward_pitch_deg;

    // Top-view: virtual camera looking straight down. The vertical half-extent
    // seen on the water equals the radius: R = H * tan(vfov/2).
    p[1].name = "topview";
    p[1].view = make_topview(s.topview_radius_m, s.topview_vfov_deg, s.topview_center_forward_m,
                             camera_height_m, out_w, out_h);

    p[2].name = "roi_zoom";
    p[2].view.yaw_deg = s.roi_yaw_deg;
    p[2].view.pitch_deg = s.roi_pitch_deg;
    p[2].view.hfov_deg = s.roi_hfov_deg;

    // Diagnostic / as-is view: the original fisheye image, unrotated (not one of the 3 spec presets).
    p[3].name = "raw_fisheye";
    p[3].view.kind = geo::kViewRaw;
    p[3].view.raw_fill = s.raw_fill ? 1 : 0;

    if (do_fit) {
        char buf[240];
        // perspective views: narrow the field of view until every pixel is real camera data
        for (int idx : {0, 2}) {
            geo::ViewParams& v = p[static_cast<std::size_t>(idx)].view;
            const double before = v.hfov_deg;
            const double frac0 = visible_fraction(*fit, v, out_w, out_h);
            while (visible_fraction(*fit, v, out_w, out_h) < kFullyVisible && v.hfov_deg > 15.0) v.hfov_deg *= 0.96;
            if (v.hfov_deg < before) {
                std::snprintf(buf, sizeof(buf), "%s: %.0f%% visible -> hfov %.0f -> %.0f deg; ",
                              p[static_cast<std::size_t>(idx)].name.c_str(), 100.0 * frac0, before, v.hfov_deg);
                note += buf;
            }
        }
        // top-view: keep the radius, move the centre forward until nothing behind the camera is shown;
        // if that is impossible shrink the radius
        double radius = s.topview_radius_m;
        bool found = false;
        double c = s.topview_center_forward_m;
        for (int shrink = 0; shrink < 14 && !found; ++shrink, radius *= 0.9) {
            for (c = s.topview_center_forward_m; c <= s.topview_center_forward_m + 4.0 * radius; c += 0.25) {
                if (visible_fraction(*fit, make_topview(radius, s.topview_vfov_deg, c, camera_height_m, out_w, out_h),
                                     out_w, out_h) >= kFullyVisible) { found = true; break; }
            }
            if (found) break;
        }
        if (found) {
            const double margin = 0.3;
            p[1].view = make_topview(radius, s.topview_vfov_deg, c + margin, camera_height_m, out_w, out_h);
            std::snprintf(buf, sizeof(buf), "topview: centre %.1f m ahead of the camera, radius %.1f m (asked %.1f m at %.1f m); ",
                          c + margin, radius, s.topview_radius_m, s.topview_center_forward_m);
            note += buf;
        } else {
            note += "topview: no fully visible window found; ";
        }
    }
    if (report) *report = note;
    return p;
}

ViewController::ViewController(std::vector<ViewPreset> presets, double transition_ms)
    : presets_(std::move(presets)), transition_ms_(std::max(0.0, transition_ms)) {
    if (!presets_.empty()) { from_ = to_ = presets_[0].view; }
}

bool ViewController::select(int index, double now_ms) {
    if (index < 0 || index >= static_cast<int>(presets_.size())) return false;
    if (started_ && index == target_) return true;
    const geo::ViewParams& dest = presets_[static_cast<std::size_t>(index)].view;
    if (!started_) {
        // First selection: show it immediately, no fly-in from nowhere.
        from_ = to_ = dest;
        t0_ms_ = now_ms - transition_ms_;
    } else {
        from_ = current(now_ms);
        to_ = dest;
        t0_ms_ = now_ms;
    }
    target_ = index;
    started_ = true;
    return true;
}

geo::ViewParams ViewController::current(double now_ms) const {
    if (!started_) return to_;
    if (transition_ms_ <= 0.0) return to_;
    const double t = (now_ms - t0_ms_) / transition_ms_;
    if (t >= 1.0) return to_;
    return lerp_view(from_, to_, smoothstep(t));
}

bool ViewController::in_transition(double now_ms) const {
    return started_ && transition_ms_ > 0.0 && (now_ms - t0_ms_) < transition_ms_;
}

} // namespace avm::view
