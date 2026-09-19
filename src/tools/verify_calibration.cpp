// verify_calibration - accuracy check of a fisheye calibration against known
// distances (FR-2.4, PR-4: Top-View distance error within +-5 % for 0-15 m).
//
//   verify_calibration --config config/avm_config.json --points measurements.csv
//       [--calib eo_calib.json] [--tolerance-pct 5] [--report report.md]
//
// measurements.csv columns: kind,label,u1,v1,u2,v2,expected_m
//   pair  : distance between two marks (e.g. bollard spacing, deck markings)
//   range : distance of mark 1 from the camera foot point
// Pixel coordinates are read from the raw fisheye image (e.g. clicked by hand).
// Exit code: 0 = all within tolerance, 1 = out of tolerance / invalid, 2 = usage.

#include "src/calib/verification.hpp"
#include "src/config/app_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace avm;

int main(int argc, char** argv) {
    std::string config_path, calib_path, points_path, report_path;
    double tol = 5.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--config") config_path = next();
        else if (a == "--calib") calib_path = next();
        else if (a == "--points") points_path = next();
        else if (a == "--tolerance-pct") tol = std::atof(next());
        else if (a == "--report") report_path = next();
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (points_path.empty()) {
        std::cerr << "usage: verify_calibration --points measurements.csv [--config avm_config.json] "
                     "[--calib eo_calib.json] [--tolerance-pct 5] [--report report.md]\n";
        return 2;
    }

    config::AppConfig app = config::default_app_config();
    std::string err;
    if (!config_path.empty() && !config::load_app_config_file(config_path, app, &err)) {
        std::cerr << "config: " << err << "\n";
        return 2;
    }
    if (!calib_path.empty() && !config::load_fisheye_calibration_file(calib_path, app.eo_intrinsics, &err)) {
        std::cerr << "calibration: " << err << "\n";
        return 2;
    }

    std::ifstream f(points_path);
    if (!f) { std::cerr << "cannot open " << points_path << "\n"; return 2; }
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<calib::Measurement> ms;
    if (!calib::parse_measurements_csv(ss.str(), ms, &err)) {
        std::cerr << points_path << ": " << err << "\n";
        return 2;
    }

    const geo::CameraPose pose = geo::make_camera_pose(app.mount.height_m, app.mount.pitch_deg, app.mount.roll_deg);
    const calib::VerificationReport rep = calib::verify(app.eo_intrinsics, pose, ms, tol);

    char title[160];
    std::snprintf(title, sizeof(title), "EO calibration verification (mount h=%.2f m, pitch=%.1f deg)",
                  app.mount.height_m, app.mount.pitch_deg);
    const std::string md = calib::report_markdown(rep, title);
    std::cout << md;
    if (!report_path.empty()) {
        std::ofstream(report_path) << md;
        std::cout << "\nreport written to " << report_path << "\n";
    }
    return rep.ok() ? 0 : 1;
}
