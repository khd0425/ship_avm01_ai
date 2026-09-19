#include "verification.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace avm::calib {

static std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

bool parse_measurements_csv(const std::string& text, std::vector<Measurement>& out, std::string* err) {
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    std::vector<Measurement> result;
    while (std::getline(in, line)) {
        ++lineno;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> f;
        std::stringstream ls(line);
        std::string cell;
        while (std::getline(ls, cell, ',')) f.push_back(trim(cell));
        if (f.size() != 7) {
            if (lineno == 1 && f.size() >= 1 && f[0] == "kind") continue;   // header
            if (err) *err = "line " + std::to_string(lineno) + ": expected 7 columns";
            return false;
        }
        if (f[0] == "kind") continue;
        if (f[0] != "pair" && f[0] != "range") {
            if (err) *err = "line " + std::to_string(lineno) + ": kind must be 'pair' or 'range'";
            return false;
        }
        Measurement m;
        m.kind = f[0];
        m.label = f[1];
        char* end = nullptr;
        double* dst[5] = {&m.u1, &m.v1, &m.u2, &m.v2, &m.expected_m};
        for (int i = 0; i < 5; ++i) {
            *dst[i] = std::strtod(f[static_cast<std::size_t>(i) + 2].c_str(), &end);
            if (end == f[static_cast<std::size_t>(i) + 2].c_str() || *end != '\0') {
                if (err) *err = "line " + std::to_string(lineno) + ": invalid number '" + f[static_cast<std::size_t>(i) + 2] + "'";
                return false;
            }
        }
        if (m.expected_m <= 0.0) {
            if (err) *err = "line " + std::to_string(lineno) + ": expected_m must be > 0";
            return false;
        }
        result.push_back(m);
    }
    out = std::move(result);
    return true;
}

VerificationReport verify(const geo::FisheyeModel& model, const geo::CameraPose& pose,
                          const std::vector<Measurement>& ms, double tolerance_pct) {
    VerificationReport rep;
    rep.tolerance_pct = tolerance_pct;
    double sum_pct = 0.0, sum_sq = 0.0;
    int n_valid = 0;
    for (const Measurement& m : ms) {
        MeasurementResult r;
        r.m = m;
        double x1, y1, x2 = 0, y2 = 0;
        bool ok = geo::pixel_to_ground(model, pose, m.u1, m.v1, x1, y1);
        if (ok && m.kind == "pair") ok = geo::pixel_to_ground(model, pose, m.u2, m.v2, x2, y2);
        if (ok) {
            r.valid = true;
            r.measured_m = (m.kind == "pair") ? std::hypot(x2 - x1, y2 - y1) : std::hypot(x1, y1);
            r.error_m = r.measured_m - m.expected_m;
            r.error_pct = 100.0 * r.error_m / m.expected_m;
            r.pass = std::fabs(r.error_pct) <= tolerance_pct;
            ++n_valid;
            sum_pct += std::fabs(r.error_pct);
            sum_sq += r.error_m * r.error_m;
            if (std::fabs(r.error_pct) > rep.max_abs_pct) rep.max_abs_pct = std::fabs(r.error_pct);
            (r.pass ? rep.pass : rep.fail)++;
        } else {
            ++rep.invalid;
        }
        rep.rows.push_back(r);
    }
    if (n_valid > 0) {
        rep.mean_abs_pct = sum_pct / n_valid;
        rep.rms_m = std::sqrt(sum_sq / n_valid);
    }
    return rep;
}

std::string report_markdown(const VerificationReport& r, const std::string& title) {
    std::ostringstream o;
    char buf[256];
    o << "# " << title << "\n\n";
    std::snprintf(buf, sizeof(buf), "- Tolerance: ±%.1f %%\n- Result: **%s** (pass %d / fail %d / invalid %d)\n"
                                    "- Max |error|: %.2f %%, mean |error|: %.2f %%, RMS: %.3f m\n\n",
                  r.tolerance_pct, r.ok() ? "PASS" : "FAIL", r.pass, r.fail, r.invalid,
                  r.max_abs_pct, r.mean_abs_pct, r.rms_m);
    o << buf;
    o << "| # | kind | label | expected [m] | measured [m] | error [m] | error [%] | result |\n"
      << "|---|------|-------|-------------:|-------------:|----------:|----------:|--------|\n";
    int i = 0;
    for (const auto& row : r.rows) {
        ++i;
        if (!row.valid) {
            std::snprintf(buf, sizeof(buf), "| %d | %s | %s | %.3f | - | - | - | INVALID (pixel not on water) |\n",
                          i, row.m.kind.c_str(), row.m.label.c_str(), row.m.expected_m);
        } else {
            std::snprintf(buf, sizeof(buf), "| %d | %s | %s | %.3f | %.3f | %+.3f | %+.2f | %s |\n", i,
                          row.m.kind.c_str(), row.m.label.c_str(), row.m.expected_m, row.measured_m,
                          row.error_m, row.error_pct, row.pass ? "PASS" : "FAIL");
        }
        o << buf;
    }
    return o.str();
}

} // namespace avm::calib
