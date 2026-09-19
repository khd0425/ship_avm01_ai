#pragma once

// Picture-in-picture layout for the IR windows (FR-3.1, FR-3.3): each window
// can be shown/hidden and moved by the operator (corner/side anchors or a
// free position from dragging).

#include <string>
#include <vector>

namespace avm::view {

enum class PipAnchor { TopLeft, TopRight, MiddleLeft, MiddleRight, BottomLeft, BottomRight, Custom };

/// Parse "top_left", "middle_right", ... (unknown -> fallback).
PipAnchor parse_anchor(const std::string& s, PipAnchor fallback = PipAnchor::TopLeft);
const char* to_string(PipAnchor a);

struct PipWindowCfg {
    bool visible{true};
    PipAnchor anchor{PipAnchor::TopLeft};
    double width_frac{0.22};      // window width as a fraction of the output width
    double margin_px{16.0};
    double custom_x{0.0};         // normalised top-left, used when anchor == Custom
    double custom_y{0.0};
};

struct Rect {
    int x{0}, y{0}, w{0}, h{0};
    bool contains(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; }
};

class PipLayout {
public:
    /// `src_aspect` = IR width/height (e.g. 640/480).
    explicit PipLayout(std::vector<PipWindowCfg> windows, double src_aspect = 4.0 / 3.0);

    std::size_t count() const { return win_.size(); }
    bool visible(std::size_t i) const { return win_.at(i).visible; }
    void set_visible(std::size_t i, bool v);
    void toggle(std::size_t i);
    void set_anchor(std::size_t i, PipAnchor a);

    /// Drag: place the window's top-left at a normalised position (clamped so the
    /// window stays fully on screen); the window becomes Custom-anchored.
    void move_to(std::size_t i, double nx, double ny, int out_w, int out_h);

    Rect rect(std::size_t i, int out_w, int out_h) const;

    /// Topmost visible window at a screen position, or -1.
    int hit_test(int px, int py, int out_w, int out_h) const;

private:
    std::vector<PipWindowCfg> win_;
    double aspect_;
};

} // namespace avm::view
