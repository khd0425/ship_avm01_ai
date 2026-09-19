#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>

/**
 * Undistortion Validation Tool
 * Tests fisheye undistortion on sample images
 */

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Fisheye Undistortion Validation Tool      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    // Default test input
    std::string input_path = "test/data/frames_input/frame_0000.png";
    if (argc > 1) {
        input_path = argv[1];
    }

    // Read input image
    cv::Mat fisheye = cv::imread(input_path);
    if (fisheye.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << input_path << std::endl;
        std::cout << "[INFO] Using default calibration for demonstration..." << std::endl;
    }

    std::cout << "\n[INPUT]" << std::endl;
    if (!fisheye.empty()) {
        std::cout << "  Path: " << input_path << std::endl;
        std::cout << "  Size: " << fisheye.cols << "x" << fisheye.rows << std::endl;
        std::cout << "  Type: " << fisheye.type() << std::endl;
    } else {
        // Create synthetic test image
        fisheye = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(100, 100, 100));
        cv::circle(fisheye, cv::Point(640, 360), 300, cv::Scalar(0, 255, 0), 3);
        cv::putText(fisheye, "Synthetic Fisheye", cv::Point(400, 360),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        std::cout << "  Created synthetic test image (720x1280)" << std::endl;
    }

    // Camera matrix (estimated)
    int w = fisheye.cols;
    int h = fisheye.rows;
    float f = (std::min(w, h) / 2.0f) / (3.14159f / 2.0f);  // ~180° FOV
    cv::Mat K = (cv::Mat_<float>(3, 3) <<
                 f, 0, w / 2.0f,
                 0, f, h / 2.0f,
                 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_32F);

    std::cout << "\n[CAMERA MODEL]" << std::endl;
    std::cout << "  Focal length: " << f << " pixels" << std::endl;
    std::cout << "  Principal point: (" << w / 2.0f << ", " << h / 2.0f << ")" << std::endl;
    std::cout << "  Distortion: Equidistant (no coefficients)" << std::endl;

    // Build undistortion maps
    std::cout << "\n[BUILDING REMAP]" << std::endl;
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat(), K, cv::Size(w, h), CV_32F, map1, map2);
    std::cout << "  Remap built successfully" << std::endl;
    std::cout << "  Map size: " << map1.size() << std::endl;

    // Apply undistortion
    std::cout << "\n[APPLYING UNDISTORTION]" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    cv::Mat undistorted;
    cv::remap(fisheye, undistorted, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "  Processing time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
    std::cout << "  Result size: " << undistorted.cols << "x" << undistorted.rows << std::endl;

    // Save output
    std::cout << "\n[OUTPUT]" << std::endl;
    std::string output_path = "test/data/undistortion_result.png";
    if (cv::imwrite(output_path, undistorted)) {
        std::cout << "  Saved: " << output_path << std::endl;
    } else {
        std::cerr << "  Failed to save output" << std::endl;
    }

    // Analysis
    std::cout << "\n[ANALYSIS]" << std::endl;
    cv::Scalar mean_in = cv::mean(fisheye);
    cv::Scalar mean_out = cv::mean(undistorted);
    std::cout << "  Input mean intensity: (" << std::setprecision(1)
              << mean_in[0] << ", " << mean_in[1] << ", " << mean_in[2] << ")" << std::endl;
    std::cout << "  Output mean intensity: (" << std::setprecision(1)
              << mean_out[0] << ", " << mean_out[1] << ", " << mean_out[2] << ")" << std::endl;

    // Check for black borders (undistorted region)
    int black_pixels = 0;
    int border_check_size = 50;
    for (int y = 0; y < border_check_size && y < undistorted.rows; y++) {
        for (int x = 0; x < border_check_size && x < undistorted.cols; x++) {
            cv::Vec3b pixel = undistorted.at<cv::Vec3b>(y, x);
            if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
                black_pixels++;
            }
        }
    }

    std::cout << "  Black pixels in top-left 50x50: " << black_pixels << "/2500" << std::endl;
    if (black_pixels > 2000) {
        std::cout << "  WARNING: Large black border detected" << std::endl;
    }

    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  ✓ Undistortion test completed             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    return 0;
}
