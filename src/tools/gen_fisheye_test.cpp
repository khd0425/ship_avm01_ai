#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <fstream>

/**
 * Synthetic Fisheye Video Generator
 * Creates test videos with known calibration for validation
 */

class FisheyeVideoGenerator {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        int num_frames = 150;
        int fps = 30;
        float fov = 180.0f;
        std::string output_path = "test/data/synthetic_fisheye.mp4";
    };

    static bool generate(const Config& cfg) {
        std::cout << "\n[FISHEYE VIDEO GENERATION]" << std::endl;
        std::cout << "  Output: " << cfg.output_path << std::endl;
        std::cout << "  Size: " << cfg.width << "x" << cfg.height << std::endl;
        std::cout << "  Frames: " << cfg.num_frames << " @ " << cfg.fps << " fps" << std::endl;
        std::cout << "  FOV: " << cfg.fov << "°" << std::endl;

        // Ensure output directory exists
        std::filesystem::path output_file(cfg.output_path);
        std::filesystem::create_directories(output_file.parent_path());

        // Setup video writer
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(cfg.output_path, fourcc, cfg.fps,
                               cv::Size(cfg.width, cfg.height), true);

        if (!writer.isOpened()) {
            std::cerr << "[ERROR] Cannot open video writer" << std::endl;
            return false;
        }

        std::cout << "\n[RENDERING]" << std::endl;

        float cx = cfg.width / 2.0f;
        float cy = cfg.height / 2.0f;
        float f = (std::min(cfg.width, cfg.height) / 2.0f) / std::tan(cfg.fov * 3.14159f / 360.0f);

        int progress_interval = std::max(1, cfg.num_frames / 10);

        for (int frame_idx = 0; frame_idx < cfg.num_frames; frame_idx++) {
            cv::Mat frame(cfg.height, cfg.width, CV_8UC3);

            // Generate synthetic pattern
            for (int y = 0; y < cfg.height; y++) {
                for (int x = 0; x < cfg.width; x++) {
                    // Polar coordinates from center
                    float dx = x - cx;
                    float dy = y - cy;
                    float r = std::sqrt(dx * dx + dy * dy);
                    float angle = std::atan2(dy, dx);

                    // Create concentric circle pattern
                    int circle_id = static_cast<int>(r / 50) % 8;
                    int radial_id = static_cast<int>((angle + 3.14159f) / (3.14159f / 4)) % 8;

                    // Color based on pattern
                    uint8_t b = (circle_id * 32) % 256;
                    uint8_t g = (radial_id * 32) % 256;
                    uint8_t r_val = ((circle_id + radial_id) * 16) % 256;

                    // Add animated gradient
                    int frame_phase = (frame_idx * 4) % 256;
                    uint8_t phase_color = static_cast<uint8_t>(
                        (std::sin(2.0f * 3.14159f * (r / 300.0f + frame_idx / 50.0f)) + 1.0f) * 127
                    );

                    frame.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uint8_t>((b + phase_color) / 2),
                        static_cast<uint8_t>((g + phase_color) / 2),
                        static_cast<uint8_t>((r_val + phase_color) / 2)
                    );
                }
            }

            // Add frame number and timestamp
            std::string frame_text = "Frame " + std::to_string(frame_idx);
            cv::putText(frame, frame_text, cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

            float timestamp = frame_idx / static_cast<float>(cfg.fps);
            std::string time_text = std::to_string(static_cast<int>(timestamp)) + "s";
            cv::putText(frame, time_text, cv::Point(30, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

            // Add calibration info overlay
            std::string calib_text = "FOV: " + std::to_string(static_cast<int>(cfg.fov)) + "°";
            cv::putText(frame, calib_text, cv::Point(cfg.width - 200, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            writer.write(frame);

            if ((frame_idx + 1) % progress_interval == 0) {
                std::cout << "  Progress: " << (frame_idx + 1) << "/" << cfg.num_frames
                          << " (" << ((frame_idx + 1) * 100 / cfg.num_frames) << "%)" << std::endl;
            }
        }

        writer.release();

        std::cout << "\n[CALIBRATION DATA]" << std::endl;
        std::cout << "  Camera matrix K:" << std::endl;
        std::cout << "    [ " << f << "    0  " << cx << " ]" << std::endl;
        std::cout << "    [  0   " << f << "  " << cy << " ]" << std::endl;
        std::cout << "    [  0    0      1  ]" << std::endl;
        std::cout << "  Distortion: [0, 0, 0, 0] (equidistant)" << std::endl;

        // Save calibration JSON
        std::string calib_path = "test/data/synthetic_fisheye_calib.json";
        saveCalibrationJSON(calib_path, cfg.width, cfg.height, f, cfg.fov);

        std::cout << "\n[OUTPUT FILES]" << std::endl;
        std::cout << "  Video: " << cfg.output_path << std::endl;
        std::cout << "  Calibration: " << calib_path << std::endl;

        return true;
    }

private:
    static void saveCalibrationJSON(const std::string& path, int w, int h, float f, float fov) {
        std::ofstream file(path);
        if (!file.is_open()) return;

        file << "{\n";
        file << "  \"width\": " << w << ",\n";
        file << "  \"height\": " << h << ",\n";
        file << "  \"focal_length\": " << f << ",\n";
        file << "  \"fov_degrees\": " << fov << ",\n";
        file << "  \"camera_matrix\": {\n";
        file << "    \"fx\": " << f << ",\n";
        file << "    \"fy\": " << f << ",\n";
        file << "    \"cx\": " << (w / 2.0f) << ",\n";
        file << "    \"cy\": " << (h / 2.0f) << "\n";
        file << "  },\n";
        file << "  \"distortion_model\": \"equidistant\",\n";
        file << "  \"distortion_coefficients\": [0, 0, 0, 0]\n";
        file << "}\n";
        file.close();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Synthetic Fisheye Video Generator         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    FisheyeVideoGenerator::Config cfg;

    // Parse arguments
    if (argc > 1) cfg.output_path = argv[1];
    if (argc > 2) cfg.width = std::atoi(argv[2]);
    if (argc > 3) cfg.height = std::atoi(argv[3]);
    if (argc > 4) cfg.num_frames = std::atoi(argv[4]);
    if (argc > 5) cfg.fov = std::atof(argv[5]);

    bool success = FisheyeVideoGenerator::generate(cfg);

    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    if (success) {
        std::cout << "║  ✓ Generation completed                   ║" << std::endl;
    } else {
        std::cout << "║  ✗ Generation failed                      ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    return success ? 0 : 1;
}
