#pragma once

// Histogram based automatic gain control for IR thermal frames (FR-3.2).
//
// Thermal sensors output 8-bit (MONO8) or 14/16-bit (MONO16) data whose useful
// range is often a small part of the full scale. The AGC computes a
// display look-up-table from the frame histogram:
//   * percentile clipping (ignore hot/cold outliers),
//   * a minimum output range so flat scenes do not amplify sensor noise,
//   * temporal smoothing of the window so the picture does not pump,
//   * optional plateau histogram equalisation for extra local contrast.
// The LUT (256 entries for 8-bit, 4096 for 16-bit indexed by v >> 4) is tiny:
// it is computed on the host from the pinned frame and applied on the GPU
// while the IR PiP window is composited.

#include <cstdint>
#include <vector>

namespace avm::imgproc {

enum class AgcMode { Linear, Plateau };

struct AgcParams {
    AgcMode mode{AgcMode::Plateau};
    double clip_low_percent{1.0};    // ignore darkest 1 %
    double clip_high_percent{1.0};   // ignore brightest 1 %
    double min_range_fraction{0.12}; // minimum window width as fraction of full scale
    double smoothing{0.85};          // 0 = no smoothing, ->1 = very slow
    double plateau_fraction{0.02};   // per-bin cap as fraction of pixels (Plateau)
    double plateau_blend{0.6};       // 0 = pure linear, 1 = pure plateau
    double max_gain{3.0};            // LUT slope limit vs. the linear stretch (noise protection)
};

class AgcController {
public:
    explicit AgcController(AgcParams p = {}) : p_(p) {}

    /// Update from a MONO8 frame; returns the 256-entry display LUT.
    const std::vector<std::uint8_t>& update8(const std::uint8_t* data, int width, int height,
                                             std::size_t pitch_bytes);

    /// Update from a MONO16 frame (values up to 65535); returns a 4096-entry
    /// LUT to be indexed with (value >> 4).
    const std::vector<std::uint8_t>& update16(const std::uint16_t* data, int width, int height,
                                              std::size_t pitch_bytes);

    const std::vector<std::uint8_t>& lut() const { return lut_; }
    double window_low() const { return low_; }
    double window_high() const { return high_; }
    void reset() { initialised_ = false; }

private:
    const std::vector<std::uint8_t>& build(const std::vector<std::uint32_t>& hist);

    AgcParams p_;
    std::vector<std::uint8_t> lut_;
    double low_{0.0}, high_{0.0};
    bool initialised_{false};
};

/// CPU reference used by tests and offline tools.
void apply_lut8(const std::uint8_t* src, int width, int height, std::size_t src_pitch,
                const std::vector<std::uint8_t>& lut, std::uint8_t* dst, std::size_t dst_pitch);

} // namespace avm::imgproc
