#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include <QMap>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <cmath>

/**
 * Fisheye Rectification Live Viewer — Qt5 UI
 * 
 * Displays real-time fisheye undistortion with 4 projection modes:
 * 1. Rectilinear - True pinhole projection (~150° FOV)
 * 2. Cylindrical - Horizontal wrap, full 190°+ field
 * 3. Equirectangular - 360° spherical panorama
 * 4. Mercator - Map projection with straight horizontals
 */

// ─── Projection Mode Definitions ────────────────────────────────────────────

enum class ProjectionMode {
    RECTILINEAR = 0,
    CYLINDRICAL = 1,
    EQUIRECTANGULAR = 2,
    MERCATOR = 3
};

static constexpr int NUM_PROJECTION_MODES = 4;
static constexpr float RECTILINEAR_CAP = 150.0f;
static constexpr float CYLINDRICAL_CAP = 200.0f;
static constexpr float EQUIRECTANGULAR_CAP = 180.0f;
static constexpr float MERCATOR_CAP = 180.0f;
static constexpr float VERTICAL_CAP = 140.0f;

// ─── Fisheye Unwrapper ─────────────────────────────────────────────────────

class FisheyeUnwrapper {
public:
    FisheyeUnwrapper(
        cv::Size in_size,
        cv::Mat K,
        cv::Mat D,
        ProjectionMode projection = ProjectionMode::RECTILINEAR,
        cv::Size out_size = cv::Size(),
        float out_hfov_deg = 110.0f,
        const std::string& coverage = "crop",
        float lens_fov_deg = 180.0f,
        float out_scale = 1.0f
    );

    /**
     * Set projection mode and rebuild maps if changed.
     */
    bool setProjection(ProjectionMode new_projection);

    /**
     * Apply rectification to input frame.
     */
    cv::Mat rectify(const cv::Mat& img);

    /**
     * Adjust field of view.
     */
    void nudgeFOV(float d_deg);

    /**
     * Cycle to next projection mode.
     */
    void cycleProjection();

    // Getters
    ProjectionMode getProjection() const { return projection; }
    cv::Size getOutputSize() const { return out_size; }
    float getOutHFOV() const { return out_hfov_deg; }
    float getOutVFOV() const { return out_vfov_deg; }
    const std::string& getCoverage() const { return coverage; }

private:
    void _build();
    void _buildMaps(float f_out);
    float _cap() const;
    float _autoHFOV() const;

    cv::Size in_size;
    cv::Mat K, D;
    ProjectionMode projection;
    std::string coverage;
    float lens_fov_deg;
    float out_scale;
    float out_hfov_deg;
    float out_vfov_deg;
    cv::Size out_size;
    cv::Mat _req_out_size_custom;

    cv::Mat map1, map2;
};

// ─── UI State Manager ──────────────────────────────────────────────────────

struct UIState {
    ProjectionMode mode = ProjectionMode::RECTILINEAR;
    bool show_labels = true;
    bool show_secondary = true;
    bool show_grid = false;
    bool show_fisheye = true;
};

// ─── Main Dashboard Window ────────────────────────────────────────────────

class DashboardWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit DashboardWindow(QWidget* parent = nullptr);
    ~DashboardWindow() = default;

    void updateDashboard(
        const cv::Mat& fisheye,
        const cv::Mat& rectified,
        const cv::Mat& secondary,
        float fps = 0.0f,
        const std::string& model = ""
    );

    const UIState& getUIState() const { return state; }
    UIState& getUIState() { return state; }

    bool shouldQuit() const { return quit_requested; }

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onProjectionToggled(int mode, bool checked);
    void onOptionToggled(const QString& key, bool checked);

private:
    void setupUI();
    void syncCheckboxes();
    static QPixmap frameToPixmap(const cv::Mat& frame);
    static cv::Mat addGridOverlay(const cv::Mat& frame, int step = 40);

    UIState state;
    bool quit_requested = false;

    // UI Components
    QLabel* header_title = nullptr;
    QLabel* header_status = nullptr;
    QLabel* left_label = nullptr;
    QLabel* top_label = nullptr;
    QLabel* bottom_label = nullptr;

    QMap<int, QCheckBox*> projection_checkboxes;
    QMap<QString, QCheckBox*> option_checkboxes;
    QLabel* help_label = nullptr;
};

// ─── Camera Capture Helper ────────────────────────────────────────────────

class CameraCapture {
public:
    CameraCapture(const std::string& device, int width, int height, int fps, const std::string& fourcc);
    ~CameraCapture();

    bool isOpened() const { return cap.isOpened(); }
    bool read(cv::Mat& frame) { return cap.read(frame); }
    int getWidth() const { return static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)); }
    int getHeight() const { return static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); }
    double getFPS() const { return cap.get(cv::CAP_PROP_FPS); }

private:
    cv::VideoCapture cap;
};

// ─── FPS Counter ──────────────────────────────────────────────────────────

class FPSMeter {
public:
    float tick();

private:
    std::vector<double> samples;
    static constexpr size_t MAX_SAMPLES = 30;
};
