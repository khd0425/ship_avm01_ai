#pragma once

// Rolling statistics for the performance requirements (PR-1 fps, PR-2 compute
// latency, PR-5 inference rate) and the on-screen status display (FR-8.3).

#include <algorithm>
#include <cstddef>
#include <vector>

namespace avm::metrics {

class RollingStats {
public:
    explicit RollingStats(std::size_t window = 120) : window_(std::max<std::size_t>(1, window)) {}

    void add(double v) {
        if (buf_.size() < window_) buf_.push_back(v);
        else buf_[next_] = v;
        next_ = (next_ + 1) % window_;
        ++total_;
    }
    std::size_t count() const { return buf_.size(); }
    std::size_t total() const { return total_; }
    double mean() const {
        if (buf_.empty()) return 0.0;
        double s = 0.0;
        for (double v : buf_) s += v;
        return s / static_cast<double>(buf_.size());
    }
    double max() const { return buf_.empty() ? 0.0 : *std::max_element(buf_.begin(), buf_.end()); }
    /// Nearest-rank percentile, p in [0,100].
    double percentile(double p) const {
        if (buf_.empty()) return 0.0;
        std::vector<double> s = buf_;
        std::sort(s.begin(), s.end());
        std::size_t rank = static_cast<std::size_t>(p / 100.0 * static_cast<double>(s.size()) + 0.999999);
        rank = std::clamp<std::size_t>(rank, 1, s.size());
        return s[rank - 1];
    }
    /// Fraction of samples above `limit` (e.g. frames over the 33 ms budget).
    double fraction_above(double limit) const {
        if (buf_.empty()) return 0.0;
        std::size_t n = 0;
        for (double v : buf_) if (v > limit) ++n;
        return static_cast<double>(n) / static_cast<double>(buf_.size());
    }

private:
    std::size_t window_;
    std::vector<double> buf_;
    std::size_t next_{0};
    std::size_t total_{0};
};

/// Frames per second from successive tick() timestamps (seconds).
class FpsCounter {
public:
    explicit FpsCounter(std::size_t window = 60) : intervals_(window) {}
    void tick(double now_s) {
        if (have_last_ && now_s > last_) intervals_.add(now_s - last_);
        last_ = now_s;
        have_last_ = true;
    }
    double fps() const {
        const double m = intervals_.mean();
        return m > 0.0 ? 1.0 / m : 0.0;
    }

private:
    RollingStats intervals_;
    double last_{0.0};
    bool have_last_{false};
};

} // namespace avm::metrics
