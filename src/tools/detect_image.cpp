// detect_image - run the detector on still images and save annotated copies.
// Used to validate a model / the decoding independently of the camera pipeline.
//
//   detect_image --model models/yolox_s.onnx --out out_dir img1.jpg img2.jpg ... [--conf 0.4]

#include "src/inference/dnn_detector.hpp"
#include "src/util/logger.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    std::string model = "models/yolox_s.onnx", out_dir = "detect_out";
    float conf = 0.4f;
    std::string arch = "yolox";
    std::vector<std::string> images;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--conf" && i + 1 < argc) conf = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--arch" && i + 1 < argc) arch = argv[++i];
        else images.push_back(a);
    }
    if (images.empty()) { std::cerr << "usage: detect_image --model m.onnx --out dir [--conf 0.4] [--arch yolox|yolov8] images...\n"; return 2; }
    std::filesystem::create_directories(out_dir);

    avm::DnnDetector det;
    avm::DnnDetector::Config cfg;
    cfg.model_path = model;
    cfg.confidence = conf;
    cfg.arch = arch;
    cfg.classes = {"person", "bollard", "fender", "quay_edge", "small_vessel", "buoy"};
    std::string err;
    if (!det.start(cfg, &err)) { std::cerr << err << "\n"; return 1; }

    for (const auto& path : images) {
        cv::Mat bgr = cv::imread(path);
        if (bgr.empty()) { std::cerr << "cannot read " << path << "\n"; continue; }
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        while (!det.submit(rgb.data, rgb.step, rgb.cols, rgb.rows)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        avm::DetectionResult res;
        while (!det.fetch(res)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::printf("%s: %zu detections in %.0f ms\n", path.c_str(), res.boxes.size(), det.last_inference_ms());
        for (const auto& b : res.boxes) {
            const cv::Rect r(static_cast<int>(b.x * bgr.cols), static_cast<int>(b.y * bgr.rows),
                             static_cast<int>(b.w * bgr.cols), static_cast<int>(b.h * bgr.rows));
            cv::rectangle(bgr, r, {0, 0, 255}, 3);
            char lab[64];
            std::snprintf(lab, sizeof(lab), "%s %.0f%%", cfg.classes[static_cast<std::size_t>(b.class_id)].c_str(), b.confidence * 100);
            cv::putText(bgr, lab, {r.x + 4, std::max(24, r.y - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {0, 0, 255}, 2);
            std::printf("   %-12s %.2f  box(%.2f,%.2f,%.2f,%.2f)\n", cfg.classes[static_cast<std::size_t>(b.class_id)].c_str(),
                        b.confidence, b.x, b.y, b.w, b.h);
        }
        cv::imwrite(out_dir + "/" + std::filesystem::path(path).stem().string() + "_det.jpg", bgr);
    }
    det.stop();
    return 0;
}
