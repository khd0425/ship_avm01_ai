#include "test_util.hpp"

#include "src/imgproc/agc.hpp"
#include "src/sync/frame_sync.hpp"

#include <vector>

using namespace avm;

static void test_sync_matching() {
    sync::SyncConfig cfg;   // 16 ms tolerance
    sync::FrameSynchronizer s(2, cfg);
    const std::uint64_t now = 1'000'000;

    // IR0 at 33.3 ms cadence, IR1 lagging by 25 ms.
    for (int i = 0; i < 5; ++i) {
        s.push_ir(0, {static_cast<std::uint64_t>(1'000'000 + i * 33'333), static_cast<std::uint32_t>(i)}, now);
        s.push_ir(1, {static_cast<std::uint64_t>(1'025'000 + i * 33'333), static_cast<std::uint32_t>(i)}, now);
    }
    s.push_eo({1'066'666, 2}, now);

    auto m = s.match(1'066'666, now);
    CHECK(m.size() == 2);
    CHECK(m[0].used);
    CHECK(m[0].stamp.frame_id == 2);
    CHECK(std::llabs(m[0].delta_us) <= 16000);
    // IR1 nearest frame is 1'058'333 (id 1): delta -8.3 ms -> used
    CHECK(m[1].used);
    CHECK(m[1].stamp.frame_id == 1);

    // EO far away from every IR frame -> both excluded (FR-1.3)
    auto far = s.match(1'500'000, now);
    CHECK(!far[0].used && !far[1].used);
    CHECK(far[0].state == sync::ChannelState::Active);   // just out of tolerance, still connected
    CHECK(s.stats(0).excluded_out_of_tolerance == 1);
    CHECK(s.stats(0).matched == 1);
    CHECK(s.stats(0).max_abs_delta_us <= 16000);

    // boundary: exactly 16 ms is accepted, 16.001 ms is not
    sync::FrameSynchronizer b(1, cfg);
    b.push_ir(0, {1'016'000, 1}, now);
    CHECK(b.match(1'000'000, now)[0].used);
    sync::FrameSynchronizer b2(1, cfg);
    b2.push_ir(0, {1'016'001, 1}, now);
    CHECK(!b2.match(1'000'000, now)[0].used);

    std::string line = sync::FrameSynchronizer::format_match_log(1'066'666, m);
    CHECK(line.find("IR0=") != std::string::npos && line.find("IR1=") != std::string::npos);
}

static void test_sync_disconnect() {
    sync::SyncConfig cfg;
    cfg.stale_timeout_us = 500'000;
    sync::FrameSynchronizer s(2, cfg);

    // Never received anything -> disconnected, pipeline keeps working.
    auto m0 = s.match(1'000'000, 1'000'000);
    CHECK(!m0[0].used && m0[0].state == sync::ChannelState::Disconnected);

    s.push_ir(0, {1'000'000, 1}, 1'000'000);
    s.push_ir(1, {1'000'000, 1}, 1'000'000);
    CHECK(s.match(1'000'000, 1'000'100)[0].used);

    // IR1 keeps streaming, IR0 stops (cable pulled): FR-1.4
    s.push_ir(1, {1'600'000, 2}, 1'600'000);
    auto m = s.match(1'600'000, 1'600'000);
    CHECK(m[0].state == sync::ChannelState::Disconnected && !m[0].used);
    CHECK(m[1].state == sync::ChannelState::Active && m[1].used);
    CHECK(s.stats(0).excluded_disconnected >= 1);

    // recovery
    s.push_ir(0, {1'633'000, 3}, 1'633'000);
    CHECK(s.match(1'633'000, 1'633'000)[0].used);

    // explicit driver notification
    s.set_link_up(1, false);
    CHECK(s.state(1, 1'633'000) == sync::ChannelState::Disconnected);
    s.set_link_up(1, true);
    CHECK(s.state(1, 1'633'000) == sync::ChannelState::Active);

    CHECK(!s.eo_connected(1'000));
    s.push_eo({1, 1}, 2'000'000);
    CHECK(s.eo_connected(2'100'000));
    CHECK(!s.eo_connected(2'700'000));

    // out-of-range channel index is ignored, not a crash
    s.push_ir(9, {1, 1}, 1);
    s.set_link_up(-1, true);
    CHECK(s.state(9, 0) == sync::ChannelState::Disconnected);
}

static std::vector<std::uint8_t> make_frame(int w, int h, int lo, int hi) {
    std::vector<std::uint8_t> f(static_cast<std::size_t>(w) * h);
    for (std::size_t i = 0; i < f.size(); ++i)
        f[i] = static_cast<std::uint8_t>(lo + (hi - lo) * static_cast<int>(i % 997) / 996);
    return f;
}

static void test_agc() {
    const int w = 64, h = 48;
    // Low-contrast thermal scene: only 20 grey levels used (FR-3.2).
    auto f = make_frame(w, h, 100, 120);

    for (auto mode : {imgproc::AgcMode::Linear, imgproc::AgcMode::Plateau}) {
        imgproc::AgcParams p;
        p.mode = mode;
        imgproc::AgcController agc(p);
        const auto& lut = agc.update8(f.data(), w, h, w);
        CHECK(lut.size() == 256);
        for (int i = 1; i < 256; ++i) CHECK(lut[i] >= lut[i - 1]);   // monotonic
        std::vector<std::uint8_t> out(f.size());
        imgproc::apply_lut8(f.data(), w, h, w, lut, out.data(), w);
        int mn = 255, mx = 0;
        for (auto v : out) { mn = std::min<int>(mn, v); mx = std::max<int>(mx, v); }
        CHECK(mx - mn >= 150);           // contrast stretched far beyond the original 20 levels
    }

    // Flat scene must not turn sensor noise into full-range contrast.
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(w) * h, 128);
    imgproc::AgcController agc;
    const auto& lut = agc.update8(flat.data(), w, h, w);
    CHECK(agc.window_high() - agc.window_low() >= 0.12 * 255 - 1.0);   // min range enforced
    int spread = std::abs(int(lut[127]) - int(lut[129]));
    CHECK(spread <= 60);                 // noise (127 vs 129) stays quiet: max_gain limits the slope

    // Temporal smoothing: window moves gradually after a scene change.
    imgproc::AgcParams sp; sp.smoothing = 0.9;
    imgproc::AgcController sm(sp);
    auto a = make_frame(w, h, 20, 60);
    auto b = make_frame(w, h, 180, 220);
    sm.update8(a.data(), w, h, w);
    double lo0 = sm.window_low();
    sm.update8(b.data(), w, h, w);
    double lo1 = sm.window_low();
    CHECK(lo1 > lo0 && lo1 < 150);       // moved toward the new scene, not jumped
    for (int i = 0; i < 100; ++i) sm.update8(b.data(), w, h, w);
    CHECK(sm.window_low() > 150);        // converges

    // 16-bit thermal: 4096-entry LUT indexed by v>>4
    std::vector<std::uint16_t> f16(static_cast<std::size_t>(w) * h);
    for (std::size_t i = 0; i < f16.size(); ++i) f16[i] = static_cast<std::uint16_t>(30000 + (i % 500));
    imgproc::AgcController a16;
    const auto& l16 = a16.update16(f16.data(), w, h, w * 2);
    CHECK(l16.size() == 4096);
    CHECK(l16[(30000 + 499) >> 4] > l16[30000 >> 4]);
}

static void test_sync_evaluate() {
    sync::SyncConfig cfg;
    sync::FrameSynchronizer s(1, cfg);
    s.push_ir(0, {1'000'000, 1}, 1'000'000);
    sync::FrameStamp near_f{1'010'000, 2}, far_f{1'030'000, 3};
    auto a = s.evaluate(0, &near_f, 1'000'000, 1'010'000);
    CHECK(a.used && a.delta_us == 10'000 && a.stamp.frame_id == 2);
    auto b = s.evaluate(0, &far_f, 1'000'000, 1'030'000);
    CHECK(!b.used && b.state == sync::ChannelState::Active);
    auto c = s.evaluate(0, nullptr, 1'000'000, 1'030'000);
    CHECK(!c.used && c.state == sync::ChannelState::Disconnected);
    auto d = s.evaluate(5, &near_f, 0, 0);                  // invalid channel
    CHECK(!d.used && d.state == sync::ChannelState::Disconnected);
    CHECK(s.stats(0).matched == 1 && s.stats(0).excluded_out_of_tolerance == 1 &&
          s.stats(0).excluded_disconnected == 1);
}

int main() {
    test_sync_evaluate();
    test_sync_matching();
    test_sync_disconnect();
    test_agc();
    TEST_MAIN_END("test_sync_agc");
}
