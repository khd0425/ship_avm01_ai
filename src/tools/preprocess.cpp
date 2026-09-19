#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>

/**
 * Image Preprocessing Utility
 * Batch processing for fisheye image collection
 */

struct PreprocessConfig {
    std::string input_dir = "test/data/frames_input";
    std::string output_dir = "test/data/frames_preprocessed";
    int target_width = 1280;
    int target_height = 720;
    float brightness_adjust = 0.0f;   // -100 to +100
    float contrast_adjust = 1.0f;     // 0.5 to 2.0
    bool equalize_hist = false;
    bool denoise = false;
};

class ImagePreprocessor {
public:
    static bool processDirectory(const PreprocessConfig& cfg) {
        std::cout << "\n[IMAGE PREPROCESSING]" << std::endl;
        std::cout << "  Input: " << cfg.input_dir << std::endl;
        std::cout << "  Output: " << cfg.output_dir << std::endl;
        std::cout << "  Target size: " << cfg.target_width << "x" << cfg.target_height << std::endl;
        std::cout << "  Brightness: " << cfg.brightness_adjust << std::endl;
        std::cout << "  Contrast: " << cfg.contrast_adjust << std::endl;

        if (!std::filesystem::exists(cfg.input_dir)) {
            std::cout << "[WARN] Input directory not found, creating test files..." << std::endl;
            createTestImages(cfg.input_dir, 5);
        }

        std::filesystem::create_directories(cfg.output_dir);

        std::vector<std::string> image_files;
        for (const auto& entry : std::filesystem::directory_iterator(cfg.input_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                    image_files.push_back(entry.path().string());
                }
            }
        }

        if (image_files.empty()) {
            std::cerr << "[ERROR] No image files found" << std::endl;
            return false;
        }

        std::cout << "\n[PROCESSING " << image_files.size() << " FILES]" << std::endl;

        int processed = 0;
        int failed = 0;

        for (size_t i = 0; i < image_files.size(); i++) {
            std::string input_path = image_files[i];
            std::string filename = std::filesystem::path(input_path).filename().string();
            std::string output_path = cfg.output_dir + "/" + filename;

            cv::Mat img = cv::imread(input_path);
            if (img.empty()) {
                std::cerr << "  FAILED: " << filename << " (cannot read)" << std::endl;
                failed++;
                continue;
            }

            // Resize
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(cfg.target_width, cfg.target_height), 0, 0, cv::INTER_LINEAR);

            // Brightness and contrast
            if (cfg.brightness_adjust != 0.0f || cfg.contrast_adjust != 1.0f) {
                resized = adjustBrightnessContrast(resized, cfg.brightness_adjust, cfg.contrast_adjust);
            }

            // Histogram equalization
            if (cfg.equalize_hist) {
                resized = equalizeHistogram(resized);
            }

            // Denoising
            if (cfg.denoise) {
                cv::fastNlMeansDenoisingColored(resized, resized, 10, 10, 15, 21);
            }

            // Save
            if (cv::imwrite(output_path, resized)) {
                std::cout << "  OK: " << filename << " (" << resized.cols << "x" << resized.rows << ")" << std::endl;
                processed++;
            } else {
                std::cerr << "  FAILED: " << filename << " (cannot write)" << std::endl;
                failed++;
            }
        }

        std::cout << "\n[RESULTS]" << std::endl;
        std::cout << "  Processed: " << processed << std::endl;
        std::cout << "  Failed: " << failed << std::endl;
        std::cout << "  Total: " << image_files.size() << std::endl;

        return failed == 0;
    }

    static void printUsage() {
        std::cout << "\nUsage: preprocess [options]" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  -i, --input <dir>       Input directory (default: test/data/frames_input)" << std::endl;
        std::cout << "  -o, --output <dir>      Output directory (default: test/data/frames_preprocessed)" << std::endl;
        std::cout << "  -w, --width <px>        Target width (default: 1280)" << std::endl;
        std::cout << "  -h, --height <px>       Target height (default: 720)" << std::endl;
        std::cout << "  -b, --brightness <val>  Brightness adjustment -100 to +100 (default: 0)" << std::endl;
        std::cout << "  -c, --contrast <val>    Contrast adjustment 0.5 to 2.0 (default: 1.0)" << std::endl;
        std::cout << "  --equalize              Apply histogram equalization" << std::endl;
        std::cout << "  --denoise               Apply denoising filter" << std::endl;
    }

private:
    static cv::Mat adjustBrightnessContrast(const cv::Mat& src, float brightness, float contrast) {
        cv::Mat dst = src.clone();
        dst.convertTo(dst, -1, contrast, brightness);
        return dst;
    }

    static cv::Mat equalizeHistogram(const cv::Mat& src) {
        if (src.channels() == 1) {
            cv::Mat dst;
            cv::equalizeHist(src, dst);
            return dst;
        } else {
            std::vector<cv::Mat> channels;
            cv::split(src, channels);
            for (auto& ch : channels) {
                cv::equalizeHist(ch, ch);
            }
            cv::Mat dst;
            cv::merge(channels, dst);
            return dst;
        }
    }

    static void createTestImages(const std::string& output_dir, int count) {
        std::filesystem::create_directories(output_dir);
        std::cout << "  Creating " << count << " test images..." << std::endl;

        for (int i = 0; i < count; i++) {
            cv::Mat img(720, 1280, CV_8UC3);

            // Generate gradient pattern
            for (int y = 0; y < 720; y++) {
                for (int x = 0; x < 1280; x++) {
                    uint8_t b = static_cast<uint8_t>((x * 255) / 1280);
                    uint8_t g = static_cast<uint8_t>((y * 255) / 720);
                    uint8_t r = static_cast<uint8_t>(((x + y) * 255) / (1280 + 720));
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
                }
            }

            // Add text
            cv::putText(img, "Test Image " + std::to_string(i), cv::Point(50, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            std::string path = output_dir + "/test_" + std::to_string(i) + ".png";
            cv::imwrite(path, img);
        }

        std::cout << "  Test images created in " << output_dir << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Image Preprocessing Utility               ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    PreprocessConfig cfg;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            cfg.input_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            cfg.target_width = std::atoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            cfg.target_height = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--brightness") && i + 1 < argc) {
            cfg.brightness_adjust = std::atof(argv[++i]);
        } else if ((arg == "-c" || arg == "--contrast") && i + 1 < argc) {
            cfg.contrast_adjust = std::atof(argv[++i]);
        } else if (arg == "--equalize") {
            cfg.equalize_hist = true;
        } else if (arg == "--denoise") {
            cfg.denoise = true;
        } else if (arg == "--help") {
            ImagePreprocessor::printUsage();
            return 0;
        }
    }

    bool success = ImagePreprocessor::processDirectory(cfg);

    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    if (success) {
        std::cout << "║  ✓ Preprocessing completed                ║" << std::endl;
    } else {
        std::cout << "║  ✗ Preprocessing with errors              ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    return success ? 0 : 1;
}
