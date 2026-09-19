#include "test_util.hpp"

#include "src/metrics/perf_stats.hpp"
#include "src/record/recording_controller.hpp"
#include "src/util/logger.hpp"
#include "src/view/pip_layout.hpp"
#include "src/view/view_controller.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace avm;

static void test_view_controller() {
    view::PresetSettings ps;
    auto presets = view::make_default_presets(ps, 10.0, 1920, 1080);
    view::ViewController vc(presets, 700.0);

    CHECK(vc.count() == 4);                                  // 3 spec presets + raw fisheye
    CHECK(vc.name(0) == "forward" && vc.name(1) == "topview" && vc.name(2) == "roi_zoom");
    CHECK(vc.name(3) == "raw_fisheye" && presets[3].view.kind == avm::geo::kViewRaw);
    CHECK(!vc.select(4, 0) && !vc.select(-1, 0));

    // FR-4.1 / FR-4.4: preset 0 shown immediately, ROI is narrower than forward
    CHECK(vc.select(0, 0.0));
    CHECK(!vc.in_transition(0.0));
    CHECK_NEAR(vc.current(0.0).hfov_deg, ps.forward_hfov_deg, 1e-9);
    CHECK(presets[2].view.hfov_deg < presets[0].view.hfov_deg);
    CHECK_NEAR(presets[1].view.pitch_deg, 90.0, 1e-9);   // top-view looks straight down

    // FR-4.5: smooth transition, monotone, no jump at the ends
    CHECK(vc.select(1, 1000.0));
    CHECK(vc.in_transition(1000.0));
    CHECK_NEAR(vc.current(1000.0).pitch_deg, presets[0].view.pitch_deg, 1e-9);   // starts where we were
    double prev = vc.current(1000.0).pitch_deg, max_step = 0.0;
    for (double t = 1001.0; t <= 1700.0; t += 1.0) {
        double p = vc.current(t).pitch_deg;
        CHECK(p >= prev - 1e-12);
        max_step = std::max(max_step, p - prev);
        prev = p;
    }
    CHECK(max_step < 0.5);                                                         // < 0.5 deg per ms
    CHECK_NEAR(vc.current(1700.0).pitch_deg, 90.0, 1e-9);
    CHECK(!vc.in_transition(1701.0));

    // Interrupting a transition continues from the displayed view (no jump).
    CHECK(vc.select(0, 2000.0));
    view::ViewController vc2(presets, 700.0);
    vc2.select(0, 0.0);
    vc2.select(1, 100.0);
    auto mid = vc2.current(450.0);
    vc2.select(2, 450.0);
    auto after = vc2.current(450.0);
    CHECK_NEAR(after.pitch_deg, mid.pitch_deg, 1e-9);
    CHECK_NEAR(after.offset_up_m, mid.offset_up_m, 1e-9);
    CHECK_NEAR(after.hfov_deg, mid.hfov_deg, 1e-9);

    // instant switch when transition_ms == 0
    view::ViewController inst(presets, 0.0);
    inst.select(0, 0.0); inst.select(1, 5.0);
    CHECK_NEAR(inst.current(5.0).pitch_deg, 90.0, 1e-9);
}

static void test_pip_layout() {
    view::PipWindowCfg l, r;
    l.anchor = view::PipAnchor::MiddleLeft;
    r.anchor = view::PipAnchor::MiddleRight;
    view::PipLayout pip({l, r}, 640.0 / 480.0);
    const int W = 1920, H = 1080;

    auto a = pip.rect(0, W, H), b = pip.rect(1, W, H);
    CHECK(a.w == 422 && a.h == 317);                        // 0.22 * 1920, 4:3
    CHECK(a.x == 16 && b.x == W - b.w - 16);
    CHECK(a.y == (H - a.h) / 2 && b.y == a.y);
    CHECK(a.x + a.w <= b.x);                                // windows do not overlap

    CHECK(pip.hit_test(a.x + 5, a.y + 5, W, H) == 0);
    CHECK(pip.hit_test(b.x + 5, b.y + 5, W, H) == 1);
    CHECK(pip.hit_test(W / 2, H / 2, W, H) == -1);

    pip.toggle(0);                                          // FR-3.3 show/hide
    CHECK(!pip.visible(0));
    CHECK(pip.hit_test(a.x + 5, a.y + 5, W, H) == -1);
    pip.set_visible(0, true);

    pip.set_anchor(0, view::PipAnchor::BottomRight);
    auto c = pip.rect(0, W, H);
    CHECK(c.x == W - c.w - 16 && c.y == H - c.h - 16);

    pip.move_to(1, 0.9, 0.95, W, H);                        // drag beyond the edge -> clamped on screen
    auto d = pip.rect(1, W, H);
    CHECK(d.x + d.w <= W && d.y + d.h <= H && d.x >= 0 && d.y >= 0);
    pip.move_to(1, -0.5, -0.5, W, H);
    auto e = pip.rect(1, W, H);
    CHECK(e.x == 0 && e.y == 0);

    CHECK(view::parse_anchor("middle_left") == view::PipAnchor::MiddleLeft);
    CHECK(view::parse_anchor("bogus", view::PipAnchor::BottomLeft) == view::PipAnchor::BottomLeft);
    CHECK(std::string(view::to_string(view::PipAnchor::TopRight)) == "top_right");
}

static void test_perf_stats() {
    metrics::RollingStats s(5);
    for (double v : {10.0, 20.0, 30.0, 40.0, 50.0}) s.add(v);
    CHECK_NEAR(s.mean(), 30.0, 1e-9);
    CHECK_NEAR(s.max(), 50.0, 1e-9);
    CHECK_NEAR(s.percentile(99), 50.0, 1e-9);
    CHECK_NEAR(s.percentile(50), 30.0, 1e-9);
    CHECK_NEAR(s.fraction_above(33.0), 0.4, 1e-9);           // 40 and 50 exceed the 33 ms budget
    s.add(100.0);                                            // window rolls over
    CHECK(s.count() == 5 && s.total() == 6);
    CHECK_NEAR(s.max(), 100.0, 1e-9);

    metrics::FpsCounter f(30);
    for (int i = 0; i < 60; ++i) f.tick(i / 30.0);
    CHECK_NEAR(f.fps(), 30.0, 1e-6);
    CHECK_NEAR(metrics::FpsCounter().fps(), 0.0, 0);
}

namespace {
struct FakeSink : record::IVideoSink {
    bool open_ok{true};
    int fail_after{-1};
    int written{0};
    bool opened{false};
    std::string path;
    bool open(const std::string& p, int, int, double) override { path = p; opened = open_ok; return open_ok; }
    bool write(const void*, std::size_t, int) override {
        if (fail_after >= 0 && written >= fail_after) return false;
        ++written;
        return true;
    }
    void close() override { opened = false; }
};
}

static void test_recording() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "avm_rec_test";
    fs::remove_all(dir);

    record::RecordingConfig cfg;
    cfg.directory = dir.string();
    cfg.min_free_mb = 1;
    auto sink = std::make_unique<FakeSink>();
    FakeSink* raw = sink.get();
    record::RecordingController rc(cfg, std::move(sink));

    unsigned char px[16] = {};
    CHECK(!rc.on_frame(px, 4, 4));                            // not recording
    CHECK(rc.start(1920, 1080, "20260919_120000"));           // FR-9.1 manual start
    CHECK(rc.is_recording());
    CHECK(fs::path(rc.current_path()).filename() == "avm_20260919_120000.mp4");
    CHECK(fs::exists(dir));
    CHECK(!rc.start(1920, 1080, "x"));                        // already recording
    for (int i = 0; i < 5; ++i) CHECK(rc.on_frame(px, 4, 4));
    CHECK(rc.frames_written() == 5 && raw->written == 5);
    rc.stop();                                                // manual stop
    CHECK(!rc.is_recording() && !raw->opened);

    raw->fail_after = 2; raw->written = 0;                    // encoder error stops the recording
    CHECK(rc.start(1920, 1080, "20260919_120100"));
    CHECK(rc.on_frame(px, 4, 4) && rc.on_frame(px, 4, 4));
    CHECK(!rc.on_frame(px, 4, 4));
    CHECK(!rc.is_recording() && rc.last_error().find("write failed") != std::string::npos);

    raw->open_ok = false;
    CHECK(!rc.start(1920, 1080, "20260919_120200"));
    CHECK(rc.last_error().find("cannot open") != std::string::npos);

    // absurd free-space requirement -> refuse to start
    record::RecordingConfig full = cfg;
    full.min_free_mb = ~0ull >> 20;
    record::RecordingController rc2(full, std::make_unique<FakeSink>());
    CHECK(!rc2.start(1920, 1080, "t"));
    CHECK(rc2.last_error().find("free disk") != std::string::npos);

    CHECK(record::local_timestamp_string().size() == 15);
    fs::remove_all(dir);
}

static void test_logger() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "avm_log_test" / "nested";
    fs::remove_all(dir.parent_path());

    auto& log = util::Logger::instance();
    log.set_console(false);
    log.set_level(util::LogLevel::Info);
    CHECK(log.open_directory(dir.string()));                  // FR-9.2: creates the directory
    AVM_LOGD("test", "hidden debug");
    AVM_LOGI("test", "hello %d", 42);
    AVM_LOGW("sync", "IR%d excluded (%.1f ms)", 1, 17.5);
    AVM_LOGE("cuda", "kernel failed: %s", "oops");
    log.close();

    auto read = [](const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); };
    const std::string all = read(log.log_path()), err = read(log.error_log_path());
    CHECK(all.find("hello 42") != std::string::npos);
    CHECK(all.find("hidden debug") == std::string::npos);     // level filter
    CHECK(all.find("[WARN ] [sync]") != std::string::npos);
    CHECK(all.find("kernel failed: oops") != std::string::npos);
    CHECK(err.find("hello 42") == std::string::npos);         // error log: WARN and above only
    CHECK(err.find("17.5 ms") != std::string::npos && err.find("oops") != std::string::npos);
    CHECK(log.error_count() >= 1);
    fs::remove_all(dir.parent_path());
    log.set_console(true);
}

int main() {
    test_view_controller();
    test_pip_layout();
    test_perf_stats();
    test_recording();
    test_logger();
    TEST_MAIN_END("test_view_misc");
}
