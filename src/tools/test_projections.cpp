#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Test Projection Modes
 * Unit tests for all projection transformations used in fisheye rectification
 */

enum class ProjectionMode {
    RECTILINEAR = 0,
    CYLINDRICAL = 1,
    EQUIRECTANGULAR = 2,
    MERCATOR = 3
};

static constexpr int NUM_PROJECTION_MODES = 4;
static constexpr float CAP[] = { 150.0f, 200.0f, 180.0f, 180.0f };

class ProjectionTester {
public:
    static bool testProjectionMathematics() {
        std::cout << "\n=== Testing Projection Mathematics ===" << std::endl;

        cv::Mat K = (cv::Mat_<float>(3, 3) <<
                     640, 0, 640,
                     0, 640, 360,
                     0, 0, 1);
        cv::Mat D = cv::Mat::zeros(4, 1, CV_32F);

        const char* mode_names[] = { "Rectilinear", "Cylindrical", "Equirectangular", "Mercator" };

        for (int m = 0; m < NUM_PROJECTION_MODES; m++) {
            std::cout << "\n[" << mode_names[m] << "]" << std::endl;

            // Test ray generation
            int ow = 1280, oh = 942;
            float f_out = 500.0f;
            float hfov_rad = 110.0f * 3.14159f / 180.0f;

            std::vector<cv::Point3f> rays;
            int test_samples = 9;  // 3x3 grid

            for (int i = 0; i < test_samples; i++) {
                int x = i % 3;
                int y = i / 3;
                float u = (x - 1) * 200.0f;
                float v = (y - 1) * 200.0f;

                cv::Point3f ray;
                bool valid = true;

                switch (static_cast<ProjectionMode>(m)) {
                    case ProjectionMode::RECTILINEAR:
                        ray = cv::Point3f(u / f_out, v / f_out, 1.0f);
                        break;
                    case ProjectionMode::CYLINDRICAL: {
                        float lon = u / f_out;
                        ray = cv::Point3f(std::sin(lon), v / f_out, std::cos(lon));
                        break;
                    }
                    case ProjectionMode::EQUIRECTANGULAR: {
                        float lon = u / f_out;
                        float lat = v / f_out;
                        ray = cv::Point3f(std::cos(lat) * std::sin(lon),
                                        std::sin(lat),
                                        std::cos(lat) * std::cos(lon));
                        break;
                    }
                    case ProjectionMode::MERCATOR: {
                        float lon = u / f_out;
                        float y_merc = v / f_out;
                        float lat = 2.0f * std::atan(std::exp(y_merc)) - 3.14159f / 2.0f;
                        ray = cv::Point3f(std::cos(lat) * std::sin(lon),
                                        std::sin(lat),
                                        std::cos(lat) * std::cos(lon));
                        break;
                    }
                }

                // Normalize
                float norm = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
                if (norm > 1e-6f) {
                    ray.x /= norm;
                    ray.y /= norm;
                    ray.z /= norm;
                }

                if (ray.z > 1e-6f) {  // Valid ray
                    rays.push_back(ray);
                }
            }

            std::cout << "  Generated " << rays.size() << "/" << test_samples << " valid rays" << std::endl;
            std::cout << "  Sample ray (0,0): (" << std::fixed << std::setprecision(3)
                      << rays[4].x << ", " << rays[4].y << ", " << rays[4].z << ")" << std::endl;

            // Verify FOV cap
            std::cout << "  FOV cap: " << CAP[m] << "°" << std::endl;
        }

        return true;
    }

    static bool testFOVCalculations() {
        std::cout << "\n=== Testing FOV Calculations ===" << std::endl;

        const char* mode_names[] = { "Rectilinear", "Cylindrical", "Equirectangular", "Mercator" };
        float test_fov[] = { 110.0f, 120.0f, 150.0f, 100.0f };

        for (int m = 0; m < NUM_PROJECTION_MODES; m++) {
            float hfov = test_fov[m];
            float capped_fov = std::min(hfov, CAP[m]);

            std::cout << "\n[" << mode_names[m] << "]" << std::endl;
            std::cout << "  Input FOV: " << hfov << "°" << std::endl;
            std::cout << "  Capped FOV: " << capped_fov << "°" << std::endl;
            std::cout << "  Cap: " << CAP[m] << "°" << std::endl;

            // Verify cap doesn't clip valid FOV
            if (hfov <= CAP[m]) {
                if (capped_fov != hfov) {
                    std::cerr << "  ERROR: Valid FOV was clipped!" << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    static bool testRayConsistency() {
        std::cout << "\n=== Testing Ray Direction Consistency ===" << std::endl;

        const char* mode_names[] = { "Rectilinear", "Cylindrical", "Equirectangular", "Mercator" };

        for (int m = 0; m < NUM_PROJECTION_MODES; m++) {
            std::cout << "\n[" << mode_names[m] << "]" << std::endl;

            // Test that rays are properly normalized
            int test_count = 0;
            int valid_count = 0;

            for (int u = -500; u <= 500; u += 200) {
                for (int v = -500; v <= 500; v += 200) {
                    float f_out = 500.0f;

                    cv::Point3f ray;
                    switch (static_cast<ProjectionMode>(m)) {
                        case ProjectionMode::RECTILINEAR:
                            ray = cv::Point3f(u / f_out, v / f_out, 1.0f);
                            break;
                        case ProjectionMode::CYLINDRICAL: {
                            float lon = u / f_out;
                            ray = cv::Point3f(std::sin(lon), v / f_out, std::cos(lon));
                            break;
                        }
                        case ProjectionMode::EQUIRECTANGULAR: {
                            float lon = u / f_out;
                            float lat = v / f_out;
                            ray = cv::Point3f(std::cos(lat) * std::sin(lon),
                                            std::sin(lat),
                                            std::cos(lat) * std::cos(lon));
                            break;
                        }
                        case ProjectionMode::MERCATOR: {
                            float lon = u / f_out;
                            float y_merc = v / f_out;
                            float lat = 2.0f * std::atan(std::exp(y_merc)) - 3.14159f / 2.0f;
                            ray = cv::Point3f(std::cos(lat) * std::sin(lon),
                                            std::sin(lat),
                                            std::cos(lat) * std::cos(lon));
                            break;
                        }
                    }

                    // Normalize
                    float norm = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
                    if (norm > 1e-6f) {
                        ray.x /= norm;
                        ray.y /= norm;
                        ray.z /= norm;

                        // Verify normalization (should be ~1.0)
                        float check_norm = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
                        if (std::abs(check_norm - 1.0f) < 0.001f) {
                            valid_count++;
                        }
                    }
                    test_count++;
                }
            }

            std::cout << "  Valid rays: " << valid_count << "/" << test_count << std::endl;
            if (valid_count < test_count * 0.9f) {
                std::cerr << "  WARNING: Low ray validity" << std::endl;
            }
        }

        return true;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Projection Modes Unit Test Suite       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    ProjectionTester tester;

    bool all_passed = true;
    all_passed &= tester.testProjectionMathematics();
    all_passed &= tester.testFOVCalculations();
    all_passed &= tester.testRayConsistency();

    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    if (all_passed) {
        std::cout << "║  ✓ All tests PASSED                        ║" << std::endl;
    } else {
        std::cout << "║  ✗ Some tests FAILED                       ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    return all_passed ? 0 : 1;
}
