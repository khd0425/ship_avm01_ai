#include "test_util.hpp"

#include "src/calib/verification.hpp"

#include <cmath>
#include <random>

using namespace avm;
using namespace avm::geo;

static FisheyeModel cam() {
    FisheyeModel m;
    m.width = 4056; m.height = 3040; m.cx = 2027.5; m.cy = 1519.5;
    m.fx = m.fy = 1520.0 / (kPi / 2.0);
    m.k1 = -0.01; m.k2 = 0.003; m.fov_deg = 185.0;
    return m;
}

static void test_csv() {
    std::vector<calib::Measurement> ms;
    std::string err;
    CHECK(calib::parse_measurements_csv(
        "# comment\nkind,label,u1,v1,u2,v2,expected_m\n"
        "pair, bollard spacing, 100.5, 200, 300, 400.25, 5.0\n"
        "range,mark A,1,2,0,0,10\n", ms, &err));
    CHECK(ms.size() == 2);
    CHECK(ms[0].label == "bollard spacing" && ms[0].kind == "pair");
    CHECK_NEAR(ms[0].v2, 400.25, 0); CHECK_NEAR(ms[1].expected_m, 10, 0);

    CHECK(!calib::parse_measurements_csv("pair,x,1,2,3\n", ms, &err));         // wrong column count
    CHECK(err.find("line 1") != std::string::npos);
    CHECK(!calib::parse_measurements_csv("foo,x,1,2,3,4,5\n", ms, &err));      // bad kind
    CHECK(!calib::parse_measurements_csv("pair,x,1,2,3,4,abc\n", ms, &err));   // bad number
    CHECK(!calib::parse_measurements_csv("pair,x,1,2,3,4,0\n", ms, &err));     // expected must be > 0
}

static void test_verify() {
    const FisheyeModel m = cam();
    const CameraPose pose = make_camera_pose(10.0, 20.0, 0.0);
    std::mt19937 rng(3);
    std::normal_distribution<double> noise(0.0, 0.4);   // sub-pixel localisation noise

    // Ground-truth marks 4..15 m from the ship, projected into the image.
    std::vector<calib::Measurement> ms;
    for (double r : {4.0, 6.0, 8.0, 10.0, 12.0, 15.0}) {
        for (double y : {-3.0, 2.0}) {
            double u, v;
            CHECK(ground_to_pixel(m, pose, r, y, u, v));
            calib::Measurement a;
            a.kind = "range"; a.label = "mark"; a.u1 = u + noise(rng); a.v1 = v + noise(rng);
            a.expected_m = std::hypot(r, y);
            ms.push_back(a);
        }
    }
    double ua, va, ub, vb;
    CHECK(ground_to_pixel(m, pose, 8.0, -2.0, ua, va));
    CHECK(ground_to_pixel(m, pose, 8.0, 3.0, ub, vb));   // 5.0 m apart across the beam
    calib::Measurement p;
    p.kind = "pair"; p.label = "beam"; p.u1 = ua; p.v1 = va; p.u2 = ub; p.v2 = vb; p.expected_m = 5.0;
    ms.push_back(p);

    auto good = calib::verify(m, pose, ms, 5.0);
    CHECK(good.ok());
    CHECK(good.pass == static_cast<int>(ms.size()) && good.fail == 0);
    CHECK(good.max_abs_pct < 2.0);
    std::printf("  synthetic check: max error %.3f %%, mean %.3f %%\n", good.max_abs_pct, good.mean_abs_pct);

    // A mount pitch that is wrong by 5 deg must be caught by the verification.
    auto bad = calib::verify(m, make_camera_pose(10.0, 25.0, 0.0), ms, 5.0);
    CHECK(!bad.ok() && bad.fail > 0);

    // A pixel above the horizon cannot be a water point.
    calib::Measurement sky;
    sky.kind = "range"; sky.label = "sky"; sky.u1 = m.cx; sky.v1 = 100; sky.expected_m = 10;
    auto inv = calib::verify(m, pose, {sky}, 5.0);
    CHECK(inv.invalid == 1 && !inv.ok());

    CHECK(!calib::verify(m, pose, {}, 5.0).ok());       // nothing measured is not a pass

    const std::string md = calib::report_markdown(good, "EO calibration verification");
    CHECK(md.find("**PASS**") != std::string::npos && md.find("| 1 | range |") != std::string::npos);
    CHECK(calib::report_markdown(bad, "x").find("**FAIL**") != std::string::npos);
    CHECK(calib::report_markdown(inv, "x").find("INVALID") != std::string::npos);
}

int main() {
    test_csv();
    test_verify();
    TEST_MAIN_END("test_verification");
}
