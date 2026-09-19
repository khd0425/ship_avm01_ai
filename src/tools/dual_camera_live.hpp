#pragma once

#include <QMainWindow>
#include <QLabel>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <chrono>

/**
 * Dual Camera Live Viewer — Real-time preview of two USB cameras
 * Validates dual EO/IR camera setup before feeding into AVM pipeline
 */

class FPSMeter {
public:
    float tick();

private:
    std::vector<double> samples;
    static constexpr size_t MAX_SAMPLES = 30;
};

class CameraCapture {
public:
    CameraCapture(int index_or_path, int width = 1280, int height = 720, 
                  int fps = 30, const std::string& fourcc = "MJPG");
    ~CameraCapture();

    bool isOpened() const { return cap.isOpened(); }
    bool read(cv::Mat& frame) { return cap.read(frame); }
    int getWidth() const { return static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)); }
    int getHeight() const { return static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); }
    double getFPS() const { return cap.get(cv::CAP_PROP_FPS); }

private:
    cv::VideoCapture cap;
};

class DualCameraWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit DualCameraWindow(QWidget* parent = nullptr);
    ~DualCameraWindow() = default;

    void updateFrames(const cv::Mat& left, const cv::Mat& right, float fps = 0.0f);
    bool shouldQuit() const { return quit_requested; }

protected:
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    static QPixmap frameToPixmap(const cv::Mat& frame);

    QLabel* left_label = nullptr;
    QLabel* right_label = nullptr;
    QLabel* status_label = nullptr;
    bool quit_requested = false;
};
