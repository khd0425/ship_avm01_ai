#include "test_util.hpp"

#include "src/config/app_config.hpp"
#include "src/util/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

using avm::util::Json;
namespace cfg = avm::config;

static void test_json() {
    Json j = Json::parse(R"({"a":1.5,"b":[true,false,null],"c":{"d":"x\ny\u00e9\ud83d\ude00"},"e":-2e2})");
    CHECK(j.is_object());
    CHECK_NEAR(j.number_or("a", 0), 1.5, 0);
    CHECK_NEAR(j.number_or("e", 0), -200.0, 0);
    CHECK(j.find("b")->size() == 3);
    CHECK(j.find("b")->items()[0].as_bool() == true);
    CHECK(j.find("b")->items()[2].is_null());
    CHECK(j.find("c")->string_or("d", "") == "x\ny\xC3\xA9\xF0\x9F\x98\x80");
    CHECK(j.find("missing") == nullptr);
    CHECK_NEAR(j.number_or("missing", 7), 7, 0);
    CHECK_NEAR(j.number_or("c", 7), 7, 0);   // wrong type -> default

    Json dup = Json::parse(R"({"k":1,"k":2})");
    CHECK_NEAR(dup.number_or("k", 0), 2, 0);  // last wins

    std::string err;
    Json out;
    const char* bad[] = {"", "{", "[1,]", "{\"a\":}", "{\"a\" 1}", "tru", "01x", "\"abc", "[1] x",
                         "{\"a\":\"\\q\"}", "[1e]", "-"};
    for (const char* b : bad) CHECK(!Json::try_parse(b, out, &err));
    CHECK(err.find("line") != std::string::npos);

    std::string deep(200, '[');
    CHECK(!Json::try_parse(deep, out, &err));   // depth guard, no stack overflow
}

static void test_defaults_match_spec() {
    cfg::AppConfig c = cfg::default_app_config();
    CHECK(cfg::validate(c).empty());
    CHECK(c.ir.size() == 2);                        // spec 3.1: IR x 2
    CHECK(c.eo.width == 4056 && c.eo.height == 3040);   // 12.33 MP (spec 3.3)
    CHECK_NEAR(c.sync.tolerance_us, 16000, 0);      // FR-1.3
    CHECK_NEAR(c.view.presets.topview_radius_m, 15.0, 0);   // FR-4.3
    CHECK(!c.blend.enabled);                        // FR-3.4 is optional
    CHECK(c.pip.enabled && c.pip.windows.size() == 2);
    CHECK_NEAR(c.inference.valid_range_m, 50.0, 0); // FR-7.3
    CHECK(c.inference.classes.size() == 6);         // phase-1 detector classes (spec: "6 classes")
    CHECK(c.inference.classes[0] == "person" && c.inference.classes[4] == "small_vessel");
    // f = r/(pi/2) for a 3040 px image circle over 180 deg (spec 3.2)
    CHECK_NEAR(c.eo_intrinsics.fx, 1520.0 / (avm::geo::kPi / 2), 1e-9);
    // spec 3.3: RAW10 3.70 Gbps and 49.3 MB per RGBA frame at 30 fps follow from these dimensions
    const double px = static_cast<double>(c.eo.width) * c.eo.height;
    CHECK_NEAR(px * 10 * 30 / 1e9, 3.70, 0.005);
    CHECK_NEAR(px * 4 / 1e6, 49.3, 0.05);
}

static void test_shipped_config_file() {
    // The config/avm_config.json we ship must parse, validate and agree with the defaults.
    cfg::AppConfig c = cfg::default_app_config();
    std::string err;
    CHECK(cfg::load_app_config_file(std::string(AVM_SOURCE_DIR) + "/config/avm_config.json", c, &err));
    for (const auto& e : cfg::validate(c)) std::fprintf(stderr, "  config problem: %s\n", e.c_str());
    CHECK(cfg::validate(c).empty());
    const cfg::AppConfig d = cfg::default_app_config();
    CHECK(c.eo.width == d.eo.width && c.eo.height == d.eo.height);
    CHECK(c.ir.size() == 2 && c.pip.windows.size() == 2);
    CHECK_NEAR(c.eo_intrinsics.cx, d.eo_intrinsics.cx, 1e-9);
    CHECK_NEAR(c.eo_intrinsics.fx, d.eo_intrinsics.fx, 0.1);
    CHECK_NEAR(c.sync.tolerance_us, 16000, 0);
    CHECK_NEAR(c.view.presets.topview_radius_m, 15.0, 0);
    CHECK(!c.blend.enabled);
    CHECK(c.inference.classes.size() == d.inference.classes.size());
}

static void test_parse_overrides() {
    cfg::AppConfig c = cfg::default_app_config();
    std::string err;
    const char* text = R"({
        "_comment": "ignored",
        "camera": {"eo": {"width": 3840, "height": 2160, "source": "/dev/video0"},
                   "ir": [{"width": 320, "height": 240}, {"pixel_format": "MONO16"}, {}]},
        "output": {"width": 1280, "height": 720},
        "calibration": {"eo_intrinsics": {"fx": 900, "k1": -0.02, "fov_deg": 190},
                        "eo_mount": {"height_m": 12.5, "pitch_deg": 30}},
        "sync": {"tolerance_ms": 20, "stale_timeout_ms": 800},
        "agc": {"mode": "linear", "smoothing": 0.5},
        "pip": {"windows": [{"anchor": "top_right", "width_frac": 0.3, "visible": false}]},
        "view": {"transition_ms": 400, "topview": {"radius_m": 20}, "stabilization": {"enabled": true}},
        "inference": {"classes": ["a", "b"], "valid_range_m": 60},
        "blending": {"enabled": true},
        "recording": {"directory": "/rec", "min_free_mb": 2048},
        "logging": {"level": "debug", "console": false}
    })";
    CHECK(cfg::parse_app_config(text, c, &err));
    CHECK(c.eo.width == 3840 && c.eo.source == "/dev/video0");
    CHECK(c.ir.size() == 3);
    CHECK(c.ir[0].width == 320 && c.ir[1].pixel_format == "MONO16");
    CHECK(c.ir_intrinsics.size() == 3);
    CHECK(c.output.width == 1280);
    CHECK_NEAR(c.eo_intrinsics.fx, 900, 0);
    CHECK_NEAR(c.eo_intrinsics.k1, -0.02, 0);
    CHECK_NEAR(c.eo_intrinsics.fy, 1520.0 / (avm::geo::kPi / 2), 1e-9);   // untouched default
    CHECK_NEAR(c.mount.height_m, 12.5, 0);
    CHECK_NEAR(c.sync.tolerance_us, 20000, 0);
    CHECK_NEAR(c.sync.stale_timeout_us, 800000, 0);
    CHECK(c.agc.mode == avm::imgproc::AgcMode::Linear);
    CHECK(c.pip.windows.size() == 1 && !c.pip.windows[0].visible);
    CHECK(c.pip.windows[0].anchor == avm::view::PipAnchor::TopRight);
    CHECK_NEAR(c.view.presets.transition_ms, 400, 0);
    CHECK_NEAR(c.view.presets.topview_radius_m, 20, 0);
    CHECK(c.view.stabilization);
    CHECK(c.inference.classes.size() == 2);
    CHECK(c.blend.enabled);
    CHECK(c.recording.directory == "/rec" && c.recording.min_free_mb == 2048);
    CHECK(!c.logging.console);
    // 3 IR but only 1 PiP window -> validation flags it
    bool flagged = false;
    for (auto& e : cfg::validate(c)) if (e.find("pip.windows") != std::string::npos) flagged = true;
    CHECK(flagged);

    cfg::AppConfig d = cfg::default_app_config();
    CHECK(!cfg::parse_app_config("{ nope", d, &err));
    CHECK(!err.empty());
    CHECK(!cfg::parse_app_config("[1,2]", d, &err));
    CHECK(!cfg::load_app_config_file("/nonexistent/avm.json", d, &err));
}

static void test_calibration_file_roundtrip() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "avm_cfg_test";
    fs::create_directories(dir);
    const std::string calib = (dir / "eo.json").string();

    avm::geo::FisheyeModel m;
    m.fx = 1001.25; m.fy = 1002.5; m.cx = 2010.5; m.cy = 1515.25;
    m.k1 = -0.0123; m.k2 = 0.004; m.k3 = -0.0007; m.k4 = 0.0001;
    m.fov_deg = 188; m.width = 4032; m.height = 3040;
    { std::ofstream(calib) << cfg::fisheye_calibration_to_json(m, 0.31, 24); }

    avm::geo::FisheyeModel r;
    std::string err;
    CHECK(cfg::load_fisheye_calibration_file(calib, r, &err));
    CHECK_NEAR(r.fx, m.fx, 1e-6); CHECK_NEAR(r.fy, m.fy, 1e-6);
    CHECK_NEAR(r.cx, m.cx, 1e-6); CHECK_NEAR(r.cy, m.cy, 1e-6);
    CHECK_NEAR(r.k1, m.k1, 1e-9); CHECK_NEAR(r.k4, m.k4, 1e-9);
    CHECK_NEAR(r.fov_deg, 188, 1e-9);
    CHECK(r.width == 4032 && r.height == 3040);

    // config can reference the calibration file
    cfg::AppConfig c = cfg::default_app_config();
    std::string text = std::string("{\"calibration\":{\"eo_intrinsics\":{\"file\":\"") + calib + "\"}}}";
    CHECK(cfg::parse_app_config(text, c, &err));
    CHECK_NEAR(c.eo_intrinsics.fx, 1001.25, 1e-6);

    { std::ofstream(dir / "bad.json") << "{\"foo\":1}"; }
    CHECK(!cfg::load_fisheye_calibration_file((dir / "bad.json").string(), r, &err));
    fs::remove_all(dir);
}

int main() {
    test_json();
    test_defaults_match_spec();
    test_shipped_config_file();
    test_parse_overrides();
    test_calibration_file_roundtrip();
    TEST_MAIN_END("test_json_config");
}
