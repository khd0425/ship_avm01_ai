#include "frame_sync.hpp"

#include <cstdio>
#include <cstdlib>

namespace avm::sync {

FrameSynchronizer::FrameSynchronizer(int num_ir_channels, SyncConfig cfg)
    : cfg_(cfg),
      hist_(static_cast<std::size_t>(num_ir_channels)),
      last_rx_us_(static_cast<std::size_t>(num_ir_channels), 0),
      ever_rx_(static_cast<std::size_t>(num_ir_channels), false),
      link_up_(static_cast<std::size_t>(num_ir_channels), true),
      stats_(static_cast<std::size_t>(num_ir_channels)) {}

void FrameSynchronizer::push_eo(const FrameStamp&, std::uint64_t now_us) {
    eo_last_rx_us_ = now_us;
    eo_ever_rx_ = true;
}

void FrameSynchronizer::push_ir(int channel, const FrameStamp& s, std::uint64_t now_us) {
    if (channel < 0 || channel >= num_ir_channels()) return;
    const auto c = static_cast<std::size_t>(channel);
    hist_[c].push_back(s);
    while (hist_[c].size() > cfg_.history) hist_[c].pop_front();
    last_rx_us_[c] = now_us;
    ever_rx_[c] = true;
}

void FrameSynchronizer::set_link_up(int channel, bool up) {
    if (channel < 0 || channel >= num_ir_channels()) return;
    link_up_[static_cast<std::size_t>(channel)] = up;
}

ChannelState FrameSynchronizer::state(int channel, std::uint64_t now_us) const {
    if (channel < 0 || channel >= num_ir_channels()) return ChannelState::Disconnected;
    const auto c = static_cast<std::size_t>(channel);
    if (!link_up_[c] || !ever_rx_[c]) return ChannelState::Disconnected;
    const std::int64_t age = static_cast<std::int64_t>(now_us) - static_cast<std::int64_t>(last_rx_us_[c]);
    return age > cfg_.stale_timeout_us ? ChannelState::Disconnected : ChannelState::Active;
}

bool FrameSynchronizer::eo_connected(std::uint64_t now_us) const {
    if (!eo_ever_rx_) return false;
    const std::int64_t age = static_cast<std::int64_t>(now_us) - static_cast<std::int64_t>(eo_last_rx_us_);
    return age <= cfg_.stale_timeout_us;
}

ChannelMatch FrameSynchronizer::evaluate(int channel, const FrameStamp* candidate,
                                         std::uint64_t eo_timestamp_us, std::uint64_t now_us) {
    ChannelMatch m;
    if (channel < 0 || channel >= num_ir_channels()) {
        m.state = ChannelState::Disconnected;
        return m;
    }
    ChannelStats& st = stats_[static_cast<std::size_t>(channel)];
    m.state = state(channel, now_us);
    if (m.state == ChannelState::Disconnected || !candidate) {
        m.used = false;
        m.state = ChannelState::Disconnected;
        ++st.excluded_disconnected;
        return m;
    }
    const std::int64_t d = static_cast<std::int64_t>(candidate->timestamp_us) -
                           static_cast<std::int64_t>(eo_timestamp_us);
    const std::int64_t a = std::llabs(d);
    m.delta_us = d;
    m.stamp = *candidate;
    if (a > cfg_.tolerance_us) {
        m.used = false;
        ++st.excluded_out_of_tolerance;
    } else {
        m.used = true;
        ++st.matched;
        if (a > st.max_abs_delta_us) st.max_abs_delta_us = a;
        st.mean_abs_delta_us += (static_cast<double>(a) - st.mean_abs_delta_us) /
                                static_cast<double>(st.matched);
    }
    return m;
}

std::vector<ChannelMatch> FrameSynchronizer::match(std::uint64_t eo_timestamp_us, std::uint64_t now_us) {
    std::vector<ChannelMatch> out;
    out.reserve(hist_.size());
    for (std::size_t c = 0; c < hist_.size(); ++c) {
        // Nearest frame in time among the remembered ones.
        const FrameStamp* best = nullptr;
        std::int64_t best_abs = 0;
        for (const FrameStamp& f : hist_[c]) {
            const std::int64_t a = std::llabs(static_cast<std::int64_t>(f.timestamp_us) -
                                              static_cast<std::int64_t>(eo_timestamp_us));
            if (!best || a < best_abs) { best = &f; best_abs = a; }
        }
        out.push_back(evaluate(static_cast<int>(c), best, eo_timestamp_us, now_us));
    }
    return out;
}

std::string FrameSynchronizer::format_match_log(std::uint64_t eo_timestamp_us,
                                                const std::vector<ChannelMatch>& m) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "sync EO_ts=%llu",
                  static_cast<unsigned long long>(eo_timestamp_us));
    std::string s = buf;
    for (std::size_t i = 0; i < m.size(); ++i) {
        if (m[i].state == ChannelState::Disconnected) {
            std::snprintf(buf, sizeof(buf), " IR%zu=DISCONNECTED", i);
        } else {
            std::snprintf(buf, sizeof(buf), " IR%zu=%+.2fms%s", i,
                          static_cast<double>(m[i].delta_us) / 1000.0,
                          m[i].used ? "" : "(EXCLUDED)");
        }
        s += buf;
    }
    return s;
}

} // namespace avm::sync
