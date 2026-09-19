#include "test_util.hpp"

#include "src/config/app_config.hpp"
#include "src/geometry/lut_builder.hpp"
#include "src/geometry/view_math.hpp"
#include "src/view/view_controller.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

using namespace avm;
using namespace avm::geo;

static FisheyeModel spec_camera() {
    // 12.33 MP fisheye from the spec: 4056x3040, image circle 3040 px = 180 deg.
    FisheyeModel m;
    m.width = 4056; m.height = 3040;
    m.cx = 2027.5; m.cy = 1519.5;
    m.fx = m.fy = 1520.0 / (kPi / 2.0);
    m.k1 = -0.01; m.k2 = 0.003; m.k3 = -0.0005; m.k4 = 0.0;
    m.fov_deg = 185.0;
    return m;
}

static void test_fisheye_roundtrip() {
    const FisheyeModel m = spec_camera();
    const auto in = to_intrinsics<double>(m);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> ang(0.0, 2 * kPi);
    std::uniform_real_distribution<double> th(0.0, 92.0 * kDegToRad);
    int ok = 0;
    for (int i = 0; i < 2000; ++i) {
        double t = th(rng), a = ang(rng);
        Vec3<double> ray{std::sin(t) * std::cos(a), std::sin(t) * std::sin(a), std::cos(t)};
        double u, v;
        if (!project_fisheye(in, ray, u, v)) continue;
        Vec3<double> back;
        CHECK(unproject_fisheye(in, u, v, back));
        CHECK_NEAR(back.x, ray.x, 1e-7); CHECK_NEAR(back.y, ray.y, 1e-7); CHECK_NEAR(back.z, ray.z, 1e-7);
        ++ok;
    }
    CHECK(ok > 1900);
    // outside the lens FOV
    double u, v;
    CHECK(!project_fisheye(in, Vec3<double>{0.0, 0.0, -1.0}, u, v));
    // optical axis -> principal point
    CHECK(project_fisheye(in, Vec3<double>{0.0, 0.0, 1.0}, u, v));
    CHECK_NEAR(u, m.cx, 1e-9); CHECK_NEAR(v, m.cy, 1e-9);
    // 90 deg off-axis lands on the image circle edge (r = f*pi/2 with k~0)
    FisheyeModel ideal = m; ideal.k1 = ideal.k2 = ideal.k3 = 0;
    CHECK(project_fisheye(to_intrinsics<double>(ideal), Vec3<double>{1.0, 0.0, 0.0}, u, v));
    CHECK_NEAR(u - ideal.cx, 1520.0, 1e-6);
}

static void test_ground_geometry_and_range_accuracy() {
    const FisheyeModel m = spec_camera();
    const CameraPose pose = make_camera_pose(10.0, 20.0, 0.0);

    // ground -> pixel -> ground round trip
    for (double x : {3.0, 5.0, 10.0, 15.0, 40.0}) {
        for (double y : {-12.0, -3.0, 0.0, 4.0, 12.0}) {
            double u = 0, v = 0, gx = 0, gy = 0;
            if (!ground_to_pixel(m, pose, x, y, u, v)) continue;
            CHECK(pixel_to_ground(m, pose, u, v, gx, gy));
            CHECK_NEAR(gx, x, 1e-6); CHECK_NEAR(gy, y, 1e-6);
        }
    }
    // The horizon direction never hits the water.
    double gx, gy;
    Vec3<double> up_ray{0.0, -0.5, 1.0};
    double uu, vv;
    CHECK(project_fisheye(to_intrinsics<double>(m), up_ray, uu, vv));
    CHECK(!pixel_to_ground(m, pose, uu, vv, gx, gy));

    // PR-4: range error < 5 % over 0-15 m for a 1 px localisation error.
    double worst = 0.0;
    for (double r = 3.0; r <= 15.0; r += 0.5) {
        double u = 0, v = 0;
        CHECK(ground_to_pixel(m, pose, r, 0.0, u, v));
        for (double du : {-1.0, 1.0}) {
            for (double dv : {-1.0, 1.0}) {
                double ex = 0, ey = 0;
                CHECK(pixel_to_ground(m, pose, u + du, v + dv, ex, ey));
                worst = std::max(worst, std::hypot(ex - r, ey) / r);
            }
        }
    }
    std::printf("  1 px error -> worst relative range error 0-15 m: %.3f %%\n", worst * 100.0);
    CHECK(worst < 0.05);
}

static void test_view_mapper_geometry() {
    const FisheyeModel m = spec_camera();
    const CameraPose pose = make_camera_pose(10.0, 20.0, 0.0);
    const int W = 1920, H = 1080;

    // Forward view co-located with the camera and looking along its axis:
    // the image centre is the principal point.
    ViewParams fwd; fwd.pitch_deg = 20.0; fwd.hfov_deg = 100.0;
    auto mf = make_view_mapper<double>(m, pose, fwd, W, H);
    double su, sv;
    CHECK(mf.map(mf.cx, mf.cy, su, sv));
    CHECK_NEAR(su, m.cx, 1e-6); CHECK_NEAR(sv, m.cy, 1e-6);
    // horizontal field of view: the image edge is hfov/2 off-axis
    Vec3<double> edge{std::tan(50.0 * kDegToRad), 0.0, 1.0};
    double eu, ev;
    CHECK(project_fisheye(to_intrinsics<double>(m), edge, eu, ev));
    CHECK(mf.map(mf.cx + W / 2.0, mf.cy, su, sv));   // image edge = half-width off-centre
    CHECK_NEAR(su, eu, 1e-6); CHECK_NEAR(sv, ev, 1e-6);

    // Rectilinear: straight lines stay straight (three collinear ground points
    // on a line parallel to the ship appear collinear in the forward view).
    ViewParams tv; tv.pitch_deg = 20.0; tv.hfov_deg = 90.0;
    auto mv = make_view_mapper<double>(m, pose, tv, W, H);
    double u1, v1, u2, v2, u3, v3;
    CHECK(mv.ground_to_view_pixel(6.0, 2.0, u1, v1));
    CHECK(mv.ground_to_view_pixel(12.0, 2.0, u2, v2));
    CHECK(mv.ground_to_view_pixel(30.0, 2.0, u3, v3));
    double cross = (u2 - u1) * (v3 - v1) - (v2 - v1) * (u3 - u1);
    CHECK(std::fabs(cross) < 1e-6 * (std::fabs(u3 - u1) + 1.0) * 1e3);

    // float (CUDA kernel precision) vs double
    auto mf32 = make_view_mapper<float>(m, pose, fwd, W, H);
    double max_diff = 0.0;
    std::mt19937 rng(7);
    for (int i = 0; i < 5000; ++i) {
        int u = rng() % W, v = rng() % H;
        double a, b; float fa, fb;
        bool r64 = mf.map(u, v, a, b);
        bool r32 = mf32.map(static_cast<float>(u), static_cast<float>(v), fa, fb);
        CHECK(r64 == r32 || std::hypot(a - m.cx, b - m.cy) > 1500);   // tolerate edge flips
        if (r64 && r32) max_diff = std::max(max_diff, std::hypot(a - fa, b - fb));
    }
    std::printf("  float vs double mapping, max diff = %.5f px\n", max_diff);
    CHECK(max_diff < 0.05);

    // Cylindrical view: horizontal axis is linear in azimuth.
    ViewParams cy; cy.kind = kViewCylindrical; cy.hfov_deg = 180.0; cy.pitch_deg = 0.0;
    FisheyeModel flat = m; flat.k1 = flat.k2 = flat.k3 = 0;
    CameraPose level = make_camera_pose(10.0, 0.0, 0.0);
    auto mc = make_view_mapper<double>(flat, level, cy, W, H);
    CHECK(mc.map(mc.cx, mc.cy, su, sv));
    CHECK_NEAR(su, flat.cx, 1e-6);
    double gxu, gyv;   // ground point at 45 deg azimuth appears at 1/4 of the width off-centre
    CHECK(mc.ground_to_view_pixel(10.0, 10.0, gxu, gyv));
    CHECK_NEAR(gxu - mc.cx, (W / 2.0) * 0.5, 1e-6);
}

static void test_raw_view() {
    // The raw preset shows the captured image as-is: no rotation, no distortion change.
    FisheyeModel m;
    m.width = 1920; m.height = 1080; m.cx = 961.5; m.cy = 507.5; m.fx = m.fy = 327; m.fov_deg = 180;
    const CameraPose pose = make_camera_pose(1.0, -60.0, 180.0);     // the mount pose must be irrelevant
    ViewParams raw; raw.kind = kViewRaw;

    auto same = make_view_mapper<double>(m, pose, raw, 1920, 1080);  // same size: exact identity
    double su, sv;
    const int pts[4][2] = {{0, 0}, {1919, 1079}, {960, 540}, {100, 900}};
    for (const auto& p : pts) {
        CHECK(same.map(p[0], p[1], su, sv));
        CHECK_NEAR(su, p[0], 1e-9); CHECK_NEAR(sv, p[1], 1e-9);
    }
    auto half = make_view_mapper<double>(m, pose, raw, 960, 540);    // half-size output: 2x scale
    CHECK(half.map(479.5, 269.5, su, sv));    CHECK_NEAR(su, 959.5, 1e-9); CHECK_NEAR(sv, 539.5, 1e-9);
    CHECK(half.map(0, 0, su, sv));            CHECK_NEAR(su, 0.5, 1e-9);
    auto tall = make_view_mapper<double>(m, pose, raw, 960, 960);    // taller output: letterboxed
    CHECK(tall.map(480, 0, su, sv));          CHECK(sv < 0.0);
    double gx, gy;                                                   // no ground geometry in this view
    CHECK(!same.view_pixel_to_ground(10, 10, gx, gy));
    CHECK(!same.ground_to_view_pixel(5, 0, gx, gy));
    auto f32 = make_view_mapper<float>(m, pose, raw, 1280, 720);     // kernel (float) precision
    float fu, fv;
    CHECK(f32.map(639.5f, 359.5f, fu, fv));   CHECK_NEAR(fu, 959.5, 1e-3);
}

static void test_fit_no_black() {
    // A single forward fisheye cannot see behind itself: the fitted views must contain no black.
    const FisheyeModel m = spec_camera();
    view::FitContext fit;
    fit.model = m; fit.src_w = m.width; fit.src_h = m.height;
    const int W = 1920, H = 1080;

    for (double tilt : {0.0, 20.0, 45.0}) {
        fit.pose = make_camera_pose(10.0, tilt, 0.0);
        view::PresetSettings ps;
        ps.fit_to_visible = false;
        auto plain = view::make_default_presets(ps, 10.0, W, H, &fit);
        // without fitting the default top-view centred on the camera is partly black
        const double plain_top = view::visible_fraction(fit, plain[1].view, W, H);
        CHECK(plain_top < 0.999);

        ps.fit_to_visible = true;
        std::string report;
        auto fitted = view::make_default_presets(ps, 10.0, W, H, &fit, &report);
        for (int i = 0; i < 4; ++i) {
            const double vis = view::visible_fraction(fit, fitted[static_cast<std::size_t>(i)].view, W, H);
            if (vis < 0.999) std::fprintf(stderr, "  tilt %.0f preset %d visible %.4f\n", tilt, i, vis);
            CHECK(vis >= 0.999);                       // every preset fully covered by real image data
        }
        // the top-view keeps the requested 15 m radius and moved forward instead
        const ViewParams& top = fitted[1].view;
        CHECK(top.offset_forward_m > plain[1].view.offset_forward_m);
        auto mt = make_view_mapper<double>(m, fit.pose, top, W, H);
        double gx0, gy0, gx1, gy1;
        CHECK(mt.view_pixel_to_ground(mt.cx, 0.0, gx0, gy0));
        CHECK(mt.view_pixel_to_ground(mt.cx, H - 1.0, gx1, gy1));
        CHECK_NEAR(gx0 - gx1, 30.0, 0.5);              // 30 m deep = radius 15 m
        CHECK(!report.empty() || tilt >= 45.0);
        std::printf("  tilt %2.0f deg: top-view centre %.1f m ahead, %s\n", tilt, top.offset_forward_m, report.c_str());
    }

    // A steeply tilted camera makes the plain forward view black at the edge; fitting narrows the FOV.
    fit.pose = make_camera_pose(1.0, -60.0, 0.0);
    view::PresetSettings ps0; ps0.fit_to_visible = false;
    auto p0 = view::make_default_presets(ps0, 1.0, W, H, &fit);
    view::PresetSettings ps1;
    auto p1 = view::make_default_presets(ps1, 1.0, W, H, &fit);
    CHECK(view::visible_fraction(fit, p0[0].view, W, H) < 0.999);
    CHECK(view::visible_fraction(fit, p1[0].view, W, H) >= 0.999);
    CHECK(p1[0].view.hfov_deg < p0[0].view.hfov_deg);
}

static void test_raw_fill() {
    // raw_fill crops to the image circle: every output pixel lies inside the circle, none is black.
    FisheyeModel m;
    m.width = 1920; m.height = 1080; m.cx = 961.5; m.cy = 507.5; m.fx = m.fy = 326.9; m.fov_deg = 180;
    const CameraPose pose = make_camera_pose(1.0, 0.0, 0.0);
    ViewParams raw; raw.kind = kViewRaw; raw.raw_fill = 1;
    const int W = 1920, H = 1080;
    auto mp = make_view_mapper<double>(m, pose, raw, W, H);
    const double r_circle = m.fx * (kPi / 2);                        // 513.5 px
    double worst = 0.0;
    for (int v : {0, H - 1}) for (int u : {0, W - 1}) {              // the output corners are the extreme points
        double su, sv;
        CHECK(mp.map(u, v, su, sv));
        worst = std::max(worst, std::hypot(su - m.cx, sv - m.cy));
    }
    CHECK(worst <= r_circle);                                        // inside the circle
    CHECK(worst > 0.9 * r_circle);                                   // and using most of it
    view::FitContext fit; fit.model = m; fit.pose = pose; fit.src_w = m.width; fit.src_h = m.height;
    CHECK(view::visible_fraction(fit, raw, W, H) >= 0.999);
    ViewParams full = raw; full.raw_fill = 0;                        // whole sensor image: black outside the circle
    CHECK(view::visible_fraction(fit, full, W, H) >= 0.999);         // (sensor pixels are valid data, just dark)
    auto mfull = make_view_mapper<double>(m, pose, full, W, H);
    double su, sv;
    CHECK(mfull.map(0, 0, su, sv));
    CHECK(std::hypot(su - m.cx, sv - m.cy) > r_circle);              // corners of the sensor are outside the circle
}

static void test_topview_ipm() {
    const FisheyeModel m = spec_camera();
    const double cam_h = 10.0;
    const CameraPose pose = make_camera_pose(cam_h, 20.0, 0.0);
    const int W = 1920, H = 1080;

    view::PresetSettings ps;
    auto presets = view::make_default_presets(ps, cam_h, W, H);
    CHECK(presets.size() == 4);   // 3 spec presets + raw fisheye
    const ViewParams& top = presets[1].view;
    auto mt = make_view_mapper<double>(m, pose, top, W, H);

    // Metric scale: R = 15 m is the vertical half-extent, so 2R/H metres per pixel.
    const double mpp = 2.0 * ps.topview_radius_m / H;
    double x0, y0, x1, y1;
    CHECK(mt.view_pixel_to_ground(mt.cx, mt.cy, x0, y0));
    CHECK_NEAR(x0, 0.0, 1e-6); CHECK_NEAR(y0, 0.0, 1e-6);
    CHECK(mt.view_pixel_to_ground(mt.cx, mt.cy - 100.0, x1, y1));   // 100 px up = forward
    CHECK_NEAR(x1, 100.0 * mpp, 1e-4); CHECK_NEAR(y1, 0.0, 1e-6);
    CHECK(mt.view_pixel_to_ground(mt.cx + 100.0, mt.cy, x1, y1));   // 100 px right = starboard
    CHECK_NEAR(y1, 100.0 * mpp, 1e-4); CHECK_NEAR(x1, 0.0, 1e-6);
    CHECK(mt.view_pixel_to_ground(mt.cx, 0.0, x1, y1));             // top edge is 15 m
    CHECK_NEAR(x1, 15.0 * (1.0 - 1.0 / H), 0.02);

    // End-to-end: a known water point lands on the right fisheye pixel.
    for (double x : {4.0, 8.0, 12.0, 15.0}) {
        for (double y : {-6.0, 0.0, 6.0}) {
            double u, v, su, sv, eu, ev;
            CHECK(mt.ground_to_view_pixel(x, y, u, v));
            CHECK(mt.map(u, v, su, sv));
            CHECK(ground_to_pixel(m, pose, x, y, eu, ev));
            CHECK_NEAR(su, eu, 1e-6); CHECK_NEAR(sv, ev, 1e-6);
        }
    }

    // LUT for the top view: the camera cannot see behind itself, so the rear
    // part is invalid (masked) while the front part is valid.
    LutMap lut = build_view_lut(m, pose, top, 480, 270, m.width, m.height);
    CHECK(lut.map_x.size() == 480u * 270u);
    double vf = lut.valid_fraction();
    CHECK(vf > 0.15 && vf < 0.95);
    CHECK(lut.map_x[static_cast<std::size_t>(20) * 480 + 240] >= 0.0f);        // ahead of the ship
    CHECK(lut.map_x[static_cast<std::size_t>(265) * 480 + 240] < 0.0f);        // far astern
}

static void test_stabilization_invariance() {
    // FR-5.1: the displayed position of a fixed water point must not depend on
    // the vessel attitude once the IMU angles are fed into the pose.
    const FisheyeModel m = spec_camera();
    const int W = 1280, H = 720;
    ViewParams fwd; fwd.pitch_deg = 15.0; fwd.hfov_deg = 90.0;

    struct Att { double pitch, roll; };
    const Att atts[] = {{0, 0}, {3, 0}, {-3, 0}, {0, 5}, {0, -5}, {2.5, -4}, {-4, 6}};
    const double gx = 9.0, gy = 2.5;

    double ref_u = 0, ref_v = 0;
    {
        CameraPose p0 = make_camera_pose(10.0, 20.0, 0.0, 0.0, 0.0);
        auto mp = make_view_mapper<double>(m, p0, fwd, W, H);
        CHECK(mp.ground_to_view_pixel(gx, gy, ref_u, ref_v));
    }
    for (const Att& a : atts) {
        CameraPose p = make_camera_pose(10.0, 20.0, 0.0, a.pitch, a.roll);
        auto mp = make_view_mapper<double>(m, p, fwd, W, H);
        double u = 0, v = 0, su = 0, sv = 0, eu = 0, ev = 0;
        CHECK(mp.ground_to_view_pixel(gx, gy, u, v));
        CHECK_NEAR(u, ref_u, 1e-9); CHECK_NEAR(v, ref_v, 1e-9);     // stabilised
        // ... and the raw pixel the camera really sees moves with the attitude:
        CHECK(mp.map(u, v, su, sv));
        CHECK(ground_to_pixel(m, p, gx, gy, eu, ev));
        CHECK_NEAR(su, eu, 1e-6); CHECK_NEAR(sv, ev, 1e-6);
    }
    // sanity: a 5 deg roll really does shift the raw pixel by many pixels
    double u0, v0, u5, v5;
    CHECK(ground_to_pixel(m, make_camera_pose(10.0, 20.0, 0.0, 0.0, 0.0), gx, gy, u0, v0));
    CHECK(ground_to_pixel(m, make_camera_pose(10.0, 20.0, 0.0, 0.0, 5.0), gx, gy, u5, v5));
    CHECK(std::hypot(u5 - u0, v5 - v0) > 50.0);
}

static void test_lut_io() {
    namespace fs = std::filesystem;
    const FisheyeModel m = spec_camera();
    const CameraPose pose = make_camera_pose(10.0, 20.0, 0.0);
    ViewParams fwd; fwd.pitch_deg = 20.0; fwd.hfov_deg = 100.0;
    LutMap lut = build_view_lut(m, pose, fwd, 96, 54, m.width, m.height);

    const fs::path dir = fs::temp_directory_path() / "avm_lut_test";
    fs::create_directories(dir);
    const std::string path = (dir / "eo_forward.avmlut").string();
    std::string err, meta;
    CHECK(save_lut(path, lut, "calib=eo_v1", &err));

    LutMap back;
    CHECK(load_lut(path, back, &meta, &err));
    CHECK(meta == "calib=eo_v1");
    CHECK(back.width == 96 && back.height == 54);
    CHECK(back.src_width == m.width && back.src_height == m.height);
    CHECK(back.map_x == lut.map_x && back.map_y == lut.map_y);

    // bit flip in the payload -> CRC mismatch
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(200);
        char c; f.seekg(200); f.get(c); f.seekp(200); f.put(static_cast<char>(c ^ 0x40));
    }
    CHECK(!load_lut(path, back, nullptr, &err));
    CHECK(err.find("CRC") != std::string::npos);

    // wrong magic / truncated / missing
    { std::ofstream(dir / "junk.bin", std::ios::binary) << "not a lut at all"; }
    CHECK(!load_lut((dir / "junk.bin").string(), back, nullptr, &err));
    fs::resize_file(path, 40);
    CHECK(!load_lut(path, back, nullptr, &err));
    CHECK(!load_lut((dir / "missing.avmlut").string(), back, nullptr, &err));
    fs::remove_all(dir);

    CHECK(crc32("123456789", 9) == 0xCBF43926u);   // standard CRC-32 check value
}

int main() {
    test_fisheye_roundtrip();
    test_ground_geometry_and_range_accuracy();
    test_view_mapper_geometry();
    test_raw_view();
    test_fit_no_black();
    test_raw_fill();
    test_topview_ipm();
    test_stabilization_invariance();
    test_lut_io();
    TEST_MAIN_END("test_geometry");
}
