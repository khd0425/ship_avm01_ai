// calib_fisheye - checkerboard based fisheye calibration (FR-2.1).
//
//   calib_fisheye --images "calib/*.jpg" --board 9x6 --square-mm 25 --out eo_calib.json [--fov 185] [--show]
//
// Estimates the Kannala-Brandt model (fx, fy, cx, cy, k1..k4) with
// cv::fisheye::calibrate and writes the JSON that config "calibration.eo_intrinsics.file",
// build_lut and verify_calibration read. Needs OpenCV (on Jetson: JetPack's OpenCV is enough).
//
// Capture 20-40 images with the board at varied positions and angles, also near
// the image border, where the fisheye distortion is strongest.

#include "src/config/app_config.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string pattern, out = "eo_calib.json";
    int bw = 9, bh = 6;
    double square_mm = 25.0, fov = 185.0;
    bool show = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--images") pattern = next();
        else if (a == "--board") { std::string b = next(); if (std::sscanf(b.c_str(), "%dx%d", &bw, &bh) != 2) { std::cerr << "--board WxH (inner corners)\n"; return 2; } }
        else if (a == "--square-mm") square_mm = std::atof(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--fov") fov = std::atof(next().c_str());
        else if (a == "--show") show = true;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (pattern.empty()) {
        std::cerr << "usage: calib_fisheye --images \"dir/*.jpg\" --board 9x6 --square-mm 25 --out eo_calib.json [--fov 185] [--show]\n";
        return 2;
    }

    std::vector<cv::String> files;
    cv::glob(pattern, files);
    if (files.empty()) { std::cerr << "no images match " << pattern << "\n"; return 1; }

    const cv::Size board(bw, bh);
    std::vector<cv::Vec3d> obj;
    for (int y = 0; y < bh; ++y)
        for (int x = 0; x < bw; ++x) obj.emplace_back(x * square_mm, y * square_mm, 0.0);

    std::vector<std::vector<cv::Vec3d>> object_points;
    std::vector<std::vector<cv::Vec2d>> image_points;
    cv::Size image_size;
    int rejected = 0;

    for (const auto& file : files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (img.empty()) { std::cerr << "skip (unreadable): " << file << "\n"; ++rejected; continue; }
        if (image_size.empty()) image_size = img.size();
        else if (img.size() != image_size) { std::cerr << "skip (size differs): " << file << "\n"; ++rejected; continue; }

        std::vector<cv::Point2f> corners;
        const bool found = cv::findChessboardCorners(
            img, board, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (!found) { std::cout << "no board: " << file << "\n"; ++rejected; continue; }

        cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
        std::vector<cv::Vec2d> pts;
        for (const auto& c : corners) pts.emplace_back(c.x, c.y);
        image_points.push_back(pts);
        object_points.push_back(obj);
        std::cout << "ok: " << file << "\n";

        if (show) {
            cv::Mat vis;
            cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(vis, board, corners, true);
            cv::imshow("corners", vis);
            cv::waitKey(200);
        }
    }

    if (image_points.size() < 10) {
        std::cerr << "only " << image_points.size() << " usable images (need >= 10, 20-40 recommended)\n";
        return 1;
    }

    cv::Matx33d K;
    cv::Vec4d D;
    std::vector<cv::Vec3d> rvecs, tvecs;
    const int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW;
    double rms = 0.0;
    try {
        rms = cv::fisheye::calibrate(object_points, image_points, image_size, K, D, rvecs, tvecs, flags,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-12));
    } catch (const cv::Exception& e) {
        std::cerr << "calibration failed: " << e.what()
                  << "\n(too few / too similar views, or a badly detected board; remove bad images and retry)\n";
        return 1;
    }

    avm::geo::FisheyeModel m;
    m.fx = K(0, 0); m.fy = K(1, 1); m.cx = K(0, 2); m.cy = K(1, 2);
    m.k1 = D[0]; m.k2 = D[1]; m.k3 = D[2]; m.k4 = D[3];
    m.fov_deg = fov;
    m.width = image_size.width; m.height = image_size.height;

    std::ofstream(out) << avm::config::fisheye_calibration_to_json(m, rms, static_cast<int>(image_points.size()));
    std::printf("calibration written to %s\n  images used %zu (rejected %d)\n  RMS reprojection error %.3f px\n"
                "  fx=%.2f fy=%.2f cx=%.2f cy=%.2f  k=[%.5f %.5f %.5f %.5f]\n",
                out.c_str(), image_points.size(), rejected, rms, m.fx, m.fy, m.cx, m.cy, m.k1, m.k2, m.k3, m.k4);
    if (rms > 1.0) std::printf("  WARNING: RMS > 1 px - check board detection, board flatness and lens focus.\n");
    return 0;
}
