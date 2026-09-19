#include "dual_camera_live.hpp"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QCloseEvent>
#include <QKeyEvent>
#include <iostream>
#include <iomanip>
#include <thread>

// ─── FPS Meter ─────────────────────────────────────────────────────────

float FPSMeter::tick() {
    auto now = std::chrono::high_resolution_clock::now();
    static auto last_time = now;

    double elapsed = std::chrono::duration<double>(now - last_time).count();
    last_time = now;

    if (elapsed > 0.0) {
        samples.push_back(1.0 / elapsed);
    }
    if (samples.size() > MAX_SAMPLES) {
        samples.erase(samples.begin());
    }

    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    return static_cast<float>(sum / std::max(size_t(1), samples.size()));
}

// ─── Camera Capture ────────────────────────────────────────────────────

CameraCapture::CameraCapture(int index, int width, int height, int fps, const std::string& fourcc)
    : cap(index, cv::CAP_V4L2)
{
    if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, fps);
        if (!fourcc.empty() && fourcc.size() == 4) {
            int cc = cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
            cap.set(cv::CAP_PROP_FOURCC, cc);
        }
    }
}

CameraCapture::~CameraCapture() {
    if (cap.isOpened()) {
        cap.release();
    }
}

// ─── Dual Camera Window ────────────────────────────────────────────────

DualCameraWindow::DualCameraWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("AVM — Dual Camera Live Viewer");
    setMinimumSize(1400, 800);

    QWidget* central = new QWidget(this);
    setCentralWidget(central);

    QVBoxLayout* main_layout = new QVBoxLayout(central);
    main_layout->setContentsMargins(12, 12, 12, 12);
    main_layout->setSpacing(12);

    // Status bar
    status_label = new QLabel("Left: 0x0 @ 0 fps | Right: 0x0 @ 0 fps");
    status_label->setStyleSheet("QLabel { color: #40b7ff; font-size: 12px; font-weight: 600; }");
    main_layout->addWidget(status_label);

    // Camera displays
    QHBoxLayout* camera_layout = new QHBoxLayout();
    camera_layout->setSpacing(16);

    left_label = new QLabel();
    left_label->setStyleSheet(
        "QLabel { background: #10141d; border: 2px solid #2f6fe5; border-radius: 12px; }"
    );
    left_label->setAlignment(Qt::AlignCenter);
    left_label->setMinimumSize(600, 500);
    left_label->setScaledContents(false);

    right_label = new QLabel();
    right_label->setStyleSheet(
        "QLabel { background: #10141d; border: 2px solid #2f6fe5; border-radius: 12px; }"
    );
    right_label->setAlignment(Qt::AlignCenter);
    right_label->setMinimumSize(600, 500);
    right_label->setScaledContents(false);

    camera_layout->addWidget(left_label);
    camera_layout->addWidget(right_label);
    main_layout->addLayout(camera_layout);

    // Help text
    QLabel* help = new QLabel("Keys: q/Esc = Quit | s = Snapshot | f = Fullscreen");
    help->setStyleSheet("QLabel { color: #98a7d9; font-size: 10px; }");
    main_layout->addWidget(help);

    setStyleSheet(
        "QMainWindow { background: #0a0e27; }\n"
        "QLabel { color: #edf4ff; }\n"
    );
}

void DualCameraWindow::updateFrames(const cv::Mat& left, const cv::Mat& right, float fps) {
    QPixmap left_pix = frameToPixmap(left);
    left_label->setPixmap(left_pix.scaled(left_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QPixmap right_pix = frameToPixmap(right);
    right_label->setPixmap(right_pix.scaled(right_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QString status = QString::asprintf(
        "Left: %dx%d @ %.1f fps | Right: %dx%d @ %.1f fps",
        left.cols, left.rows, fps, right.cols, right.rows, fps
    );
    status_label->setText(status);
}

QPixmap DualCameraWindow::frameToPixmap(const cv::Mat& frame) {
    cv::Mat display = frame.clone();
    if (display.empty()) {
        display = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::putText(display, "NO SIGNAL", cv::Point(140, 240), cv::FONT_HERSHEY_SIMPLEX,
                    1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

    if (display.channels() == 1) {
        cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    }

    cv::Mat rgb;
    cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);

    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img);
}

void DualCameraWindow::closeEvent(QCloseEvent* event) {
    quit_requested = true;
    event->accept();
}

void DualCameraWindow::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Q || event->key() == Qt::Key_Escape) {
        quit_requested = true;
    } else if (event->key() == Qt::Key_S) {
        std::cout << "[SNAP] Snapshot feature not yet implemented" << std::endl;
    } else if (event->key() == Qt::Key_F) {
        if (isFullScreen()) {
            showNormal();
        } else {
            showFullScreen();
        }
    }
    QMainWindow::keyPressEvent(event);
}

// ─── Main Entry Point ────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Parse arguments (simplified)
    int left_index = 0, right_index = 2;
    int width = 1280, height = 720, fps = 30;

    std::cout << "[INFO] Opening dual cameras: left=" << left_index << " right=" << right_index << std::endl;

    CameraCapture left_cap(left_index, width, height, fps, "MJPG");
    if (!left_cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open left camera at index " << left_index << std::endl;
        return 1;
    }

    CameraCapture right_cap(right_index, width, height, fps, "MJPG");
    if (!right_cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open right camera at index " << right_index << std::endl;
        return 1;
    }

    std::cout << "[OK] /dev/video" << left_index << ": " << left_cap.getWidth() << "x"
              << left_cap.getHeight() << " @ " << left_cap.getFPS() << " fps" << std::endl;
    std::cout << "[OK] /dev/video" << right_index << ": " << right_cap.getWidth() << "x"
              << right_cap.getHeight() << " @ " << right_cap.getFPS() << " fps" << std::endl;

    DualCameraWindow window;
    window.show();

    FPSMeter meter;
    cv::Mat left_frame, right_frame;

    while (!window.shouldQuit()) {
        bool left_ok = left_cap.read(left_frame);
        bool right_ok = right_cap.read(right_frame);

        if (!left_ok || !right_ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        float fps = meter.tick();
        window.updateFrames(left_frame, right_frame, fps);

        QApplication::processEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    std::cout << "[DONE] Dual camera viewer closed" << std::endl;
    return 0;
}
