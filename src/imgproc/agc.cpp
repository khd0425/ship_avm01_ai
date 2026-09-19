#include "agc.hpp"

#include <algorithm>
#include <cmath>

namespace avm::imgproc {

static double percentile_bin(const std::vector<std::uint32_t>& hist, std::uint64_t total, double pct) {
    const double target = total * pct / 100.0;
    double acc = 0.0;
    for (std::size_t i = 0; i < hist.size(); ++i) {
        acc += hist[i];
        if (acc >= target) return static_cast<double>(i);
    }
    return static_cast<double>(hist.size() - 1);
}

const std::vector<std::uint8_t>& AgcController::build(const std::vector<std::uint32_t>& hist) {
    const std::size_t bins = hist.size();
    std::uint64_t total = 0;
    for (auto h : hist) total += h;
    if (total == 0) {
        lut_.assign(bins, 0);
        return lut_;
    }

    double lo = percentile_bin(hist, total, p_.clip_low_percent);
    double hi = percentile_bin(hist, total, 100.0 - p_.clip_high_percent);

    // Enforce a minimum window around the centre (noise protection).
    const double min_w = std::max(2.0, p_.min_range_fraction * static_cast<double>(bins - 1));
    if (hi - lo < min_w) {
        const double c = 0.5 * (hi + lo);
        lo = c - 0.5 * min_w;
        hi = c + 0.5 * min_w;
        if (lo < 0.0) { hi -= lo; lo = 0.0; }
        if (hi > static_cast<double>(bins - 1)) { lo -= hi - static_cast<double>(bins - 1); hi = static_cast<double>(bins - 1); }
        lo = std::max(lo, 0.0);
    }

    // Temporal smoothing of the window.
    if (!initialised_) {
        low_ = lo; high_ = hi; initialised_ = true;
    } else {
        low_ = p_.smoothing * low_ + (1.0 - p_.smoothing) * lo;
        high_ = p_.smoothing * high_ + (1.0 - p_.smoothing) * hi;
    }
    const double w = std::max(1.0, high_ - low_);

    // Linear stretch of the window.
    std::vector<double> lin(bins);
    for (std::size_t i = 0; i < bins; ++i) {
        lin[i] = std::clamp((static_cast<double>(i) - low_) / w, 0.0, 1.0);
    }

    std::vector<double> out = lin;
    if (p_.mode == AgcMode::Plateau) {
        // Plateau histogram equalisation restricted to the window.
        const double cap = std::max(1.0, p_.plateau_fraction * static_cast<double>(total));
        std::vector<double> cdf(bins, 0.0);
        double acc = 0.0;
        for (std::size_t i = 0; i < bins; ++i) {
            const double in_window = (static_cast<double>(i) >= low_ && static_cast<double>(i) <= high_) ? 1.0 : 0.0;
            acc += std::min<double>(hist[i], cap) * in_window;
            cdf[i] = acc;
        }
        const double norm = acc > 0.0 ? acc : 1.0;
        for (std::size_t i = 0; i < bins; ++i) {
            const double phe = cdf[i] / norm;
            out[i] = (1.0 - p_.plateau_blend) * lin[i] + p_.plateau_blend * phe;
        }
    }

    // Slope limit: equalisation must never amplify small differences (sensor
    // noise in a flat scene) more than max_gain x the plain linear stretch.
    const double max_step = p_.max_gain / w;
    for (std::size_t i = 1; i < bins; ++i) out[i] = std::min(out[i], out[i - 1] + max_step);

    lut_.resize(bins);
    std::uint8_t prev = 0;
    for (std::size_t i = 0; i < bins; ++i) {
        auto v = static_cast<std::uint8_t>(std::lround(std::clamp(out[i], 0.0, 1.0) * 255.0));
        v = std::max(v, prev);      // guarantee a monotonic LUT
        lut_[i] = prev = v;
    }
    return lut_;
}

const std::vector<std::uint8_t>& AgcController::update8(const std::uint8_t* data, int width, int height,
                                                        std::size_t pitch_bytes) {
    std::vector<std::uint32_t> hist(256, 0);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = data + static_cast<std::size_t>(y) * pitch_bytes;
        for (int x = 0; x < width; ++x) ++hist[row[x]];
    }
    return build(hist);
}

const std::vector<std::uint8_t>& AgcController::update16(const std::uint16_t* data, int width, int height,
                                                         std::size_t pitch_bytes) {
    std::vector<std::uint32_t> hist(4096, 0);
    for (int y = 0; y < height; ++y) {
        const auto* row = reinterpret_cast<const std::uint16_t*>(
            reinterpret_cast<const std::uint8_t*>(data) + static_cast<std::size_t>(y) * pitch_bytes);
        for (int x = 0; x < width; ++x) ++hist[row[x] >> 4];
    }
    return build(hist);
}

void apply_lut8(const std::uint8_t* src, int width, int height, std::size_t src_pitch,
                const std::vector<std::uint8_t>& lut, std::uint8_t* dst, std::size_t dst_pitch) {
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* s = src + static_cast<std::size_t>(y) * src_pitch;
        std::uint8_t* d = dst + static_cast<std::size_t>(y) * dst_pitch;
        for (int x = 0; x < width; ++x) d[x] = lut[s[x]];
    }
}

} // namespace avm::imgproc
