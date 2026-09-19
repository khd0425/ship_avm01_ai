// build_lut - generate a remap LUT file (FR-2.2) from a calibration.
//
//   build_lut --config config/avm_config.json --preset 0 --out lut/eo_forward.avmlut
//   build_lut --calib eo_calib.json --height-m 10 --pitch-deg 20 --preset 1 --out lut/topview.avmlut
//
// Presets: 0 forward perspective, 1 top-view (water plane), 2 ROI zoom.
// Pure C++ (no OpenCV/CUDA): runs on the dev PC or on the Jetson.

#include "src/config/app_config.hpp"
#include "src/geometry/lut_builder.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace avm;

static void usage() {
    std::cerr <<
        "usage: build_lut --out <file.avmlut> [--config avm_config.json] [--calib eo_calib.json]\n"
        "                 [--preset 0|1|2] [--out-w N] [--out-h N]\n"
        "                 [--height-m H] [--pitch-deg P] [--roll-deg R]\n";
}

int main(int argc, char** argv) {
    std::string out, config_path, calib_path;
    int preset = 0, out_w = 0, out_h = 0;
    double height = -1, pitch = -1e9, roll = -1e9;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--out") out = next("--out");
        else if (a == "--config") config_path = next("--config");
        else if (a == "--calib") calib_path = next("--calib");
        else if (a == "--preset") preset = std::atoi(next("--preset"));
        else if (a == "--out-w") out_w = std::atoi(next("--out-w"));
        else if (a == "--out-h") out_h = std::atoi(next("--out-h"));
        else if (a == "--height-m") height = std::atof(next("--height-m"));
        else if (a == "--pitch-deg") pitch = std::atof(next("--pitch-deg"));
        else if (a == "--roll-deg") roll = std::atof(next("--roll-deg"));
        else { usage(); return 2; }
    }
    if (out.empty()) { usage(); return 2; }

    config::AppConfig app = config::default_app_config();
    std::string err;
    if (!config_path.empty() && !config::load_app_config_file(config_path, app, &err)) {
        std::cerr << "config: " << err << "\n";
        return 1;
    }
    if (!calib_path.empty() && !config::load_fisheye_calibration_file(calib_path, app.eo_intrinsics, &err)) {
        std::cerr << "calibration: " << err << "\n";
        return 1;
    }
    if (height > 0) app.mount.height_m = height;
    if (pitch > -1e8) app.mount.pitch_deg = pitch;
    if (roll > -1e8) app.mount.roll_deg = roll;
    if (out_w <= 0) out_w = static_cast<int>(app.output.width);
    if (out_h <= 0) out_h = static_cast<int>(app.output.height);

    const auto presets = view::make_default_presets(app.view.presets, app.mount.height_m, out_w, out_h);
    if (preset < 0 || preset >= static_cast<int>(presets.size())) {
        std::cerr << "preset must be 0..2\n";
        return 2;
    }

    const int src_w = app.eo_intrinsics.width > 0 ? app.eo_intrinsics.width : static_cast<int>(app.eo.width);
    const int src_h = app.eo_intrinsics.height > 0 ? app.eo_intrinsics.height : static_cast<int>(app.eo.height);
    const geo::CameraPose pose = geo::make_camera_pose(app.mount.height_m, app.mount.pitch_deg, app.mount.roll_deg);

    const auto t0 = std::chrono::steady_clock::now();
    geo::LutMap lut = geo::build_view_lut(app.eo_intrinsics, pose, presets[static_cast<std::size_t>(preset)].view,
                                          out_w, out_h, src_w, src_h);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    char meta[256];
    std::snprintf(meta, sizeof(meta), "preset=%s;height_m=%.3f;pitch_deg=%.3f;roll_deg=%.3f;fx=%.4f;fy=%.4f",
                  presets[static_cast<std::size_t>(preset)].name.c_str(), app.mount.height_m, app.mount.pitch_deg,
                  app.mount.roll_deg, app.eo_intrinsics.fx, app.eo_intrinsics.fy);
    if (!geo::save_lut(out, lut, meta, &err)) {
        std::cerr << "cannot write LUT: " << err << "\n";
        return 1;
    }
    std::printf("LUT '%s' (%s): %dx%d from %dx%d, %.1f %% of pixels valid, built in %.0f ms\n", out.c_str(),
                presets[static_cast<std::size_t>(preset)].name.c_str(), out_w, out_h, src_w, src_h,
                100.0 * lut.valid_fraction(), ms);
    return 0;
}
