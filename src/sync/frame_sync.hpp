#pragma once

// Time synchronisation of the EO and IR channels (FR-1.2, FR-1.3, FR-1.4).
//
//  * EO is the master clock. For every EO frame the nearest IR frame of each
//    IR channel is selected; if it is further away than `tolerance_us`
//    (default 16 ms) the channel is excluded from that frame.
//  * A channel with no frame for `stale_timeout_us` (cable pulled, driver
//    error) is reported as Disconnected. The pipeline keeps running and the
//    UI shows the channel as inactive.
//
// Pure host code: timestamps are supplied by the caller (capture thread).

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace avm::sync {

struct SyncConfig {
    std::int64_t tolerance_us{16000};        // FR-1.3 default 16 ms
    std::int64_t stale_timeout_us{500000};   // no frame for 0.5 s => disconnected
    std::size_t history{8};                  // frames remembered per IR channel
};

struct FrameStamp {
    std::uint64_t timestamp_us{0};   // sensor/capture time
    std::uint32_t frame_id{0};
};

enum class ChannelState { Active, Disconnected };

struct ChannelMatch {
    bool used{false};              // false => exclude this IR from the frame
    ChannelState state{ChannelState::Active};
    FrameStamp stamp{};            // matched IR frame (valid if used)
    std::int64_t delta_us{0};      // ir - eo (signed)
};

struct ChannelStats {
    std::uint64_t matched{0};
    std::uint64_t excluded_out_of_tolerance{0};
    std::uint64_t excluded_disconnected{0};
    std::int64_t max_abs_delta_us{0};
    double mean_abs_delta_us{0.0};
};

class FrameSynchronizer {
public:
    FrameSynchronizer(int num_ir_channels, SyncConfig cfg = {});

    /// Record an arrived frame. `now_us` is the local monotonic time of arrival.
    void push_eo(const FrameStamp& s, std::uint64_t now_us);
    void push_ir(int channel, const FrameStamp& s, std::uint64_t now_us);

    /// Driver-level connect/disconnect notification (e.g. V4L2 error).
    void set_link_up(int channel, bool up);

    /// Select IR frames for an EO frame and update statistics.
    std::vector<ChannelMatch> match(std::uint64_t eo_timestamp_us, std::uint64_t now_us);

    /// Judge ONE candidate IR frame (e.g. the newest one the capture ring can
    /// still provide) against an EO timestamp: applies the disconnect state and
    /// the tolerance, and updates the statistics. `candidate` may be null.
    ChannelMatch evaluate(int channel, const FrameStamp* candidate,
                          std::uint64_t eo_timestamp_us, std::uint64_t now_us);

    ChannelState state(int channel, std::uint64_t now_us) const;
    bool eo_connected(std::uint64_t now_us) const;
    const ChannelStats& stats(int channel) const { return stats_.at(static_cast<std::size_t>(channel)); }
    int num_ir_channels() const { return static_cast<int>(hist_.size()); }

    /// One-line log record with per-channel delta ("sync EO=123 IR0=+2.1ms ...").
    static std::string format_match_log(std::uint64_t eo_timestamp_us,
                                        const std::vector<ChannelMatch>& m);

private:
    SyncConfig cfg_;
    std::vector<std::deque<FrameStamp>> hist_;
    std::vector<std::uint64_t> last_rx_us_;
    std::vector<bool> ever_rx_;
    std::vector<bool> link_up_;
    std::vector<ChannelStats> stats_;
    std::uint64_t eo_last_rx_us_{0};
    bool eo_ever_rx_{false};
};

} // namespace avm::sync
