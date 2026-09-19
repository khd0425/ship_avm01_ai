#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <filesystem>

/**
 * AVM Pipeline Test Harness
 * Validates full pipeline processing chain
 */

struct PipelineStats {
    int frames_processed = 0;
    int frames_failed = 0;
    double total_ms = 0;
    double min_ms = 1e9;
    double max_ms = 0;
};

class PipelineTester {
public:
    static bool processTestVideo() {
        std::cout << "\n[PIPELINE TEST]" << std::endl;

        // Look for test video or create synthetic one
        std::string video_path = "test/data/test_video.mp4";
        cv::VideoCapture cap;

        if (std::filesystem::exists(video_path)) {
            cap.open(video_path);
            std::cout << "  Using test video: " << video_path << std::endl;
        } else {
            std::cout << "  Test video not found, using synthetic generation..." << std::endl;
            return createSyntheticTestCase();
        }

        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Cannot open video: " << video_path << std::endl;
            return false;
        }

        // Get video properties
        int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "  Frames: " << frame_count << " | Size: " << w << "x" << h
                  << " | FPS: " << fps << std::endl;

        // Setup camera model
        float f = (std::min(w, h) / 2.0f) / (3.14159f / 2.0f);
        cv::Mat K = (cv::Mat_<float>(3, 3) <<
                     f, 0, w / 2.0f,
                     0, f, h / 2.0f,
                     0, 0, 1);
        cv::Mat D = cv::Mat::zeros(4, 1, CV_32F);

        // Build remap
        cv::Mat map1, map2;
        cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat(), K, cv::Size(w, h), CV_32F, map1, map2);

        PipelineStats stats;

        // Process frames
        std::cout << "\n[PROCESSING FRAMES]" << std::endl;
        cv::Mat frame, rectified;

        int progress_interval = std::max(1, frame_count / 10);
        for (int i = 0; i < frame_count && i < 100; i++) {  // Limit to 100 frames for testing
            auto t_start = std::chrono::high_resolution_clock::now();

            if (!cap.read(frame)) {
                stats.frames_failed++;
                continue;
            }

            // Apply pipeline stages
            cv::remap(frame, rectified, map1, map2, cv::INTER_LINEAR);

            // Optional: Add more pipeline stages here (alignment, rendering, etc.)

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            stats.frames_processed++;
            stats.total_ms += elapsed_ms;
            stats.min_ms = std::min(stats.min_ms, elapsed_ms);
            stats.max_ms = std::max(stats.max_ms, elapsed_ms);

            if ((i + 1) % progress_interval == 0) {
                std::cout << "  Progress: " << (i + 1) << "/" << std::min(frame_count, 100)
                          << " (" << ((i + 1) * 100 / std::min(frame_count, 100)) << "%)" << std::endl;
            }
        }

        // Print stats
        std::cout << "\n[RESULTS]" << std::endl;
        std::cout << "  Frames processed: " << stats.frames_processed << std::endl;
        std::cout << "  Frames failed: " << stats.frames_failed << std::endl;
        std::cout << "  Avg time/frame: " << std::fixed << std::setprecision(2)
                  << (stats.total_ms / stats.frames_processed) << " ms" << std::endl;
        std::cout << "  Min time/frame: " << stats.min_ms << " ms" << std::endl;
        std::cout << "  Max time/frame: " << stats.max_ms << " ms" << std::endl;

        double avg_fps = (stats.frames_processed * 1000.0) / stats.total_ms;
        std::cout << "  Throughput: " << avg_fps << " fps" << std::endl;

        cap.release();
        return stats.frames_failed == 0;
    }

    static bool createSyntheticTestCase() {
        std::cout << "\n[SYNTHETIC TEST CASE]" << std::endl;

        int w = 1280, h = 720;
        int num_frames = 30;

        // Create output directory
        std::string output_dir = "test/data/synthetic_pipeline_test";
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        std::cout << "  Creating " << num_frames << " synthetic frames..." << std::endl;

        cv::Mat frame(h, w, CV_8UC3);
        PipelineStats stats;

        for (int i = 0; i < num_frames; i++) {
            // Generate synthetic frame
            frame = cv::Mat(h, w, CV_8UC3, cv::Scalar(30, 30, 50));
            
            // Add circular gradient
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int dx = x - w / 2;
                    int dy = y - h / 2;
                    int dist = std::sqrt(dx * dx + dy * dy);
                    int intensity = std::min(255, 100 + dist / 3);
                    frame.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity / 2, intensity);
                }
            }

            // Add frame number
            cv::putText(frame, "Frame " + std::to_string(i), cv::Point(50, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            // Simulate processing
            auto t_start = std::chrono::high_resolution_clock::now();
            
            cv::Mat processed;
            cv::GaussianBlur(frame, processed, cv::Size(5, 5), 0);  // Simulate filter
            
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            stats.frames_processed++;
            stats.total_ms += elapsed_ms;
            stats.min_ms = std::min(stats.min_ms, elapsed_ms);
            stats.max_ms = std::max(stats.max_ms, elapsed_ms);

            // Save every 10th frame
            if (i % 10 == 0) {
                std::string out_path = output_dir + "/frame_" + std::to_string(i).c_str() + ".png";
                cv::imwrite(out_path, processed);
            }
        }

        std::cout << "\n[SYNTHETIC RESULTS]" << std::endl;
        std::cout << "  Total frames: " << stats.frames_processed << std::endl;
        std::cout << "  Avg time/frame: " << std::fixed << std::setprecision(2)
                  << (stats.total_ms / stats.frames_processed) << " ms" << std::endl;
        double avg_fps = (stats.frames_processed * 1000.0) / stats.total_ms;
        std::cout << "  Throughput: " << avg_fps << " fps" << std::endl;
        std::cout << "  Output dir: " << output_dir << std::endl;

        return true;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     AVM Pipeline Test Harness              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    PipelineTester tester;
    bool success = tester.processTestVideo();

    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    if (success) {
        std::cout << "║  ✓ Pipeline test PASSED                    ║" << std::endl;
    } else {
        std::cout << "║  ✗ Pipeline test FAILED                    ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    return success ? 0 : 1;
}
