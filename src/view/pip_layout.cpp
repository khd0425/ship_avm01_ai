#include "pip_layout.hpp"

#include <algorithm>
#include <cmath>

namespace avm::view {

PipAnchor parse_anchor(const std::string& s, PipAnchor fallback) {
    if (s == "top_left") return PipAnchor::TopLeft;
    if (s == "top_right") return PipAnchor::TopRight;
    if (s == "middle_left") return PipAnchor::MiddleLeft;
    if (s == "middle_right") return PipAnchor::MiddleRight;
    if (s == "bottom_left") return PipAnchor::BottomLeft;
    if (s == "bottom_right") return PipAnchor::BottomRight;
    if (s == "custom") return PipAnchor::Custom;
    return fallback;
}

const char* to_string(PipAnchor a) {
    switch (a) {
        case PipAnchor::TopLeft: return "top_left";
        case PipAnchor::TopRight: return "top_right";
        case PipAnchor::MiddleLeft: return "middle_left";
        case PipAnchor::MiddleRight: return "middle_right";
        case PipAnchor::BottomLeft: return "bottom_left";
        case PipAnchor::BottomRight: return "bottom_right";
        case PipAnchor::Custom: return "custom";
    }
    return "top_left";
}

PipLayout::PipLayout(std::vector<PipWindowCfg> windows, double src_aspect)
    : win_(std::move(windows)), aspect_(src_aspect > 0.0 ? src_aspect : 4.0 / 3.0) {}

void PipLayout::set_visible(std::size_t i, bool v) { if (i < win_.size()) win_[i].visible = v; }
void PipLayout::toggle(std::size_t i) { if (i < win_.size()) win_[i].visible = !win_[i].visible; }
void PipLayout::set_anchor(std::size_t i, PipAnchor a) { if (i < win_.size()) win_[i].anchor = a; }

Rect PipLayout::rect(std::size_t i, int out_w, int out_h) const {
    const PipWindowCfg& c = win_.at(i);
    Rect r;
    r.w = std::max(1, static_cast<int>(std::lround(c.width_frac * out_w)));
    r.h = std::max(1, static_cast<int>(std::lround(r.w / aspect_)));
    const int m = static_cast<int>(std::lround(c.margin_px));
    const int right = out_w - r.w - m;
    const int bottom = out_h - r.h - m;
    const int mid_y = (out_h - r.h) / 2;
    switch (c.anchor) {
        case PipAnchor::TopLeft:      r.x = m;     r.y = m;      break;
        case PipAnchor::TopRight:     r.x = right; r.y = m;      break;
        case PipAnchor::MiddleLeft:   r.x = m;     r.y = mid_y;  break;
        case PipAnchor::MiddleRight:  r.x = right; r.y = mid_y;  break;
        case PipAnchor::BottomLeft:   r.x = m;     r.y = bottom; break;
        case PipAnchor::BottomRight:  r.x = right; r.y = bottom; break;
        case PipAnchor::Custom:
            r.x = static_cast<int>(std::lround(c.custom_x * out_w));
            r.y = static_cast<int>(std::lround(c.custom_y * out_h));
            break;
    }
    r.x = std::clamp(r.x, 0, std::max(0, out_w - r.w));
    r.y = std::clamp(r.y, 0, std::max(0, out_h - r.h));
    return r;
}

void PipLayout::move_to(std::size_t i, double nx, double ny, int out_w, int out_h) {
    if (i >= win_.size()) return;
    PipWindowCfg& c = win_[i];
    c.anchor = PipAnchor::Custom;
    c.custom_x = nx;
    c.custom_y = ny;
    Rect r = rect(i, out_w, out_h);          // clamped
    c.custom_x = static_cast<double>(r.x) / out_w;
    c.custom_y = static_cast<double>(r.y) / out_h;
}

int PipLayout::hit_test(int px, int py, int out_w, int out_h) const {
    for (std::size_t k = win_.size(); k-- > 0;) {          // later windows draw on top
        if (win_[k].visible && rect(k, out_w, out_h).contains(px, py)) return static_cast<int>(k);
    }
    return -1;
}

} // namespace avm::view
