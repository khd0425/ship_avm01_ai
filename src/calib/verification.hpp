#pragma once

// Calibration accuracy verification (FR-2.4, PR-4): compares distances measured
// from fisheye pixels (through the calibrated model + camera pose, projected on
// the water plane) with known ground-truth distances.

#include "src/geometry/view_math.hpp"

#include <string>
#include <vector>

namespace avm::calib {

struct Measurement {
    std::string kind;      // "pair": distance between two marked points; "range": distance of point 1 from the camera foot
    std::string label;
    double u1{0}, v1{0}, u2{0}, v2{0};   // fisheye pixel coordinates
    double expected_m{0};
};

struct MeasurementResult {
    Measurement m;
    bool valid{false};       // false if a pixel does not hit the water plane
    double measured_m{0};
    double error_m{0};
    double error_pct{0};
    bool pass{false};
};

struct VerificationReport {
    std::vector<MeasurementResult> rows;
    double tolerance_pct{5.0};
    int pass{0}, fail{0}, invalid{0};
    double max_abs_pct{0}, mean_abs_pct{0}, rms_m{0};
    bool ok() const { return !rows.empty() && fail == 0 && invalid == 0; }
};

/// CSV columns: kind,label,u1,v1,u2,v2,expected_m   ('#' comments and a header line are ignored)
bool parse_measurements_csv(const std::string& text, std::vector<Measurement>& out,
                            std::string* err = nullptr);

VerificationReport verify(const geo::FisheyeModel& model, const geo::CameraPose& pose,
                          const std::vector<Measurement>& ms, double tolerance_pct = 5.0);

std::string report_markdown(const VerificationReport& r, const std::string& title);

} // namespace avm::calib
