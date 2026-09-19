#include "fisheye_rectify_live.hpp"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QGroupBox>
#include <QSpacerItem>
#include <QCloseEvent>
#include <QTimer>
#include <QThread>
#include <iostream>
#include <iomanip>
#include <chrono>

// ─── FisheyeUnwrapper Implementation ────────────────────────────────────

FisheyeUnwrapper::FisheyeUnwrapper(
    cv::Size in_size,
    cv::Mat K,
    cv::Mat D,
    ProjectionMode projection,
    cv::Size out_size,
    float out_hfov_deg,
    const std::string& coverage,
    float lens_fov_deg,
    float out_scale
)
    : in_size(in_size),
      K(K.clone()),
      D(D.clone()),
      projection(projection),
      coverage(coverage),
      lens_fov_deg(lens_fov_deg),
      out_scale(out_scale),
      out_hfov_deg(out_hfov_deg)
{
    _build();
}

bool FisheyeUnwrapper::setProjection(ProjectionMode new_projection) {
    if (new_projection != projection) {
        projection = new_projection;
        coverage = "full";
        _build();
        return true;
    }
    return false;
}

void FisheyeUnwrapper::cycleProjection() {
    int idx = static_cast<int>(projection);
    idx = (idx + 1) % NUM_PROJECTION_MODES;
    projection = static_cast<ProjectionMode>(idx);
    coverage = "full";
    _build();
}

void FisheyeUnwrapper::nudgeFOV(float d_deg) {
    coverage = "crop";
    float cap = _cap();
    out_hfov_deg = std::min(cap, std::max(20.0f, out_hfov_deg + d_deg));
    _build();
}

float FisheyeUnwrapper::_cap() const {
    switch (projection) {
        case ProjectionMode::RECTILINEAR: return RECTILINEAR_CAP;
        case ProjectionMode::CYLINDRICAL: return CYLINDRICAL_CAP;
        case ProjectionMode::EQUIRECTANGULAR: return EQUIRECTANGULAR_CAP;
        case ProjectionMode::MERCATOR: return MERCATOR_CAP;
        default: return CYLINDRICAL_CAP;
    }
}

float FisheyeUnwrapper::_autoHFOV() const {
    float lens = lens_fov_deg > 0 ? lens_fov_deg : _cap();
    return std::min(lens, _cap());
}

void FisheyeUnwrapper::_build() {
    int w = in_size.width;
    int h = in_size.height;

    if (coverage == "full") {
        out_hfov_deg = _autoHFOV();
    }
    out_hfov_deg = std::min(_cap(), std::max(20.0f, out_hfov_deg));

    float hfov = static_cast<float>(CV_PI) * out_hfov_deg / 180.0f;

    int ow = std::max(2, static_cast<int>(std::round(w * out_scale)));
    float vfov = static_cast<float>(CV_PI) * std::min(lens_fov_deg > 0 ? lens_fov_deg : VERTICAL_CAP, VERTICAL_CAP) / 180.0f;

    int oh;
    if (projection == ProjectionMode::RECTILINEAR) {
        oh = 2.0f * (ow / 2.0f) / std::tan(hfov / 2.0f) * std::tan(vfov / 2.0f);
    } else {
        float f_cyl = (ow / 2.0f) / (hfov / 2.0f);
        oh = 2.0f * f_cyl * std::tan(vfov / 2.0f);
    }
    oh = std::max(2, std::min(static_cast<int>(std::round(oh)), static_cast<int>(std::round(w * out_scale))));
    out_size = cv::Size(ow, oh);

    float f_out;
    if (projection == ProjectionMode::RECTILINEAR) {
        f_out = (ow / 2.0f) / std::tan(hfov / 2.0f);
        out_vfov_deg = 2.0f * 180.0f / static_cast<float>(CV_PI) * std::atan((oh / 2.0f) / f_out);
    } else {
        f_out = (ow / 2.0f) / (hfov / 2.0f);
        out_vfov_deg = 2.0f * 180.0f / static_cast<float>(CV_PI) * std::atan((oh / 2.0f) / f_out);
    }

    _buildMaps(f_out);
}

void FisheyeUnwrapper::_buildMaps(float f_out) {
    int ow = out_size.width;
    int oh = out_size.height;

    // Build 3D ray directions per output pixel based on projection mode
    std::vector<cv::Point3f> pts_3d(static_cast<size_t>(oh) * ow);
    std::vector<bool> behind(static_cast<size_t>(oh) * ow, false);

    for (int y = 0; y < oh; y++) {
        float vv = y - oh / 2.0f;
        for (int x = 0; x < ow; x++) {
            float uu = x - ow / 2.0f;
            size_t idx = static_cast<size_t>(y) * ow + x;
            cv::Point3f ray;

            if (projection == ProjectionMode::RECTILINEAR) {
                ray = cv::Point3f(uu / f_out, vv / f_out, 1.0f);
            }
            else if (projection == ProjectionMode::CYLINDRICAL) {
                float lon = uu / f_out;
                ray = cv::Point3f(std::sin(lon), vv / f_out, std::cos(lon));
            }
            else if (projection == ProjectionMode::EQUIRECTANGULAR) {
                float lon = uu / f_out;
                float lat = vv / f_out;
                ray = cv::Point3f(std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon));
            }
            else { // MERCATOR
                float lon = uu / f_out;
                float y_merc = vv / f_out;
                float lat = 2.0f * std::atan(std::exp(y_merc)) - static_cast<float>(CV_PI) / 2.0f;
                ray = cv::Point3f(std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon));
            }

            float norm = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
            norm = std::max(norm, 1e-9f);
            ray.x /= norm;
            ray.y /= norm;
            ray.z /= norm;

            behind[idx] = ray.z <= 1e-6f;
            pts_3d[idx] = ray;
        }
    }

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    std::vector<cv::Point2f> img_pts;
    cv::fisheye::projectPoints(pts_3d, img_pts, rvec, tvec, K, D);

    cv::Mat img_pts_mat(oh, ow, CV_32FC2);
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            size_t idx = static_cast<size_t>(y) * ow + x;
            img_pts_mat.at<cv::Vec2f>(y, x) = behind[idx]
                ? cv::Vec2f(-1.0f, -1.0f)
                : cv::Vec2f(img_pts[idx].x, img_pts[idx].y);
        }
    }

    cv::convertMaps(img_pts_mat, cv::Mat(), map1, map2, CV_16SC2);
}

cv::Mat FisheyeUnwrapper::rectify(const cv::Mat& img) {
    cv::Mat result;
    cv::remap(img, result, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return result;
}

// ─── FPS Meter Implementation ──────────────────────────────────────────────

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

// ─── Camera Capture Implementation ────────────────────────────────────────

CameraCapture::CameraCapture(const std::string& device, int width, int height, int fps, const std::string& fourcc)
    : cap(device, cv::CAP_V4L2)
{
    if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, fps);
        if (!fourcc.empty()) {
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

// ─── Dashboard Window Implementation ────────────────────────────────────

DashboardWindow::DashboardWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("AVM — Fisheye Rectification Live");
    setMinimumSize(1400, 820);

    setupUI();
    setStyleSheet(R"(
        QMainWindow { background: #0a0e27; }
        QLabel { color: #edf4ff; }
        QCheckBox { color: #edf4ff; font-size: 13px; spacing: 10px; padding: 4px 0; }
        QCheckBox::indicator { width: 16px; height: 16px; border: 1.5px solid #78c3ff; 
                              border-radius: 4px; background: #101725; }
        QCheckBox::indicator:checked { 
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                       stop:0 #40b7ff, stop:1 #1f7ff7); 
            border: 1.5px solid #9be6ff; }
        QGroupBox { color: #78c3ff; border: 1px solid #2d3c5d; border-radius: 8px;
                   padding-top: 10px; margin-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
    )");
}

void DashboardWindow::setupUI() {
    QWidget* central = new QWidget(this);
    setCentralWidget(central);

    QVBoxLayout* root_layout = new QVBoxLayout(central);
    root_layout->setContentsMargins(18, 18, 18, 18);
    root_layout->setSpacing(14);

    // Header
    QWidget* header = new QWidget();
    QHBoxLayout* header_layout = new QHBoxLayout(header);
    header_layout->setContentsMargins(0, 0, 0, 0);

    header_title = new QLabel("AVM Live Monitor");
    header_title->setStyleSheet("QLabel { color: #40b7ff; font-size: 24px; font-weight: 700; }");

    header_status = new QLabel("Mode: RECTILINEAR | FPS: 0.0 | Labels: ON | Video2: ON");
    header_status->setStyleSheet("QLabel { color: #98a7d9; font-size: 12px; }");
    header_status->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    header_layout->addWidget(header_title);
    header_layout->addStretch();
    header_layout->addWidget(header_status);
    root_layout->addWidget(header);

    // Body: Images + Controls
    QHBoxLayout* body_row = new QHBoxLayout();
    body_row->setSpacing(16);

    // Left: Fisheye
    left_label = new QLabel();
    left_label->setStyleSheet("QLabel { background: #10141d; border: 1px solid #2f6fe5; border-radius: 12px; }");
    left_label->setAlignment(Qt::AlignCenter);
    left_label->setMinimumSize(700, 520);
    left_label->setScaledContents(false);

    // Right: Rectified + Secondary
    QWidget* right_stack = new QWidget();
    QVBoxLayout* right_layout = new QVBoxLayout(right_stack);
    right_layout->setContentsMargins(0, 0, 0, 0);
    right_layout->setSpacing(16);

    top_label = new QLabel();
    top_label->setStyleSheet("QLabel { background: #10141d; border: 1px solid #2f6fe5; border-radius: 12px; }");
    top_label->setAlignment(Qt::AlignCenter);
    top_label->setMinimumHeight(320);

    bottom_label = new QLabel();
    bottom_label->setStyleSheet("QLabel { background: #10141d; border: 1px solid #2f6fe5; border-radius: 12px; }");
    bottom_label->setAlignment(Qt::AlignCenter);
    bottom_label->setMinimumHeight(220);

    right_layout->addWidget(top_label);
    right_layout->addWidget(bottom_label);

    // Controls Panel
    QWidget* controls = new QWidget();
    controls->setFixedWidth(265);
    controls->setStyleSheet(
        "QWidget { background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, "
        "stop:0 #121a2c, stop:1 #0f1729); border: 1px solid #2d3c5d; border-radius: 12px; }"
    );
    QVBoxLayout* controls_layout = new QVBoxLayout(controls);
    controls_layout->setContentsMargins(16, 16, 16, 16);
    controls_layout->setSpacing(12);

    QLabel* title = new QLabel("Controls");
    title->setStyleSheet("QLabel { color: #edf4ff; font-size: 18px; font-weight: 700; }");
    controls_layout->addWidget(title);

    // Projection Mode Group
    QLabel* proj_label = new QLabel("Projection Mode");
    proj_label->setStyleSheet("QLabel { color: #78c3ff; font-size: 12px; font-weight: 600; margin-top: 8px; }");
    controls_layout->addWidget(proj_label);

    const char* modes[] = { "Rectilinear", "Cylindrical", "Equirectangular", "Mercator" };
    for (int i = 0; i < NUM_PROJECTION_MODES; i++) {
        QCheckBox* box = new QCheckBox(modes[i]);
        box->setChecked(i == 0);
        connect(box, QOverload<bool>::of(&QCheckBox::toggled), this,
                [this, i](bool checked) { onProjectionToggled(i, checked); });
        projection_checkboxes[i] = box;
        controls_layout->addWidget(box);
    }

    // Options Group
    QLabel* opt_label = new QLabel("Options");
    opt_label->setStyleSheet("QLabel { color: #78c3ff; font-size: 12px; font-weight: 600; margin-top: 8px; }");
    controls_layout->addWidget(opt_label);

    const char* options[] = { "Labels", "Video2", "Grid" };
    const char* opt_keys[] = { "labels", "video2", "grid" };
    bool opt_defaults[] = { true, true, false };

    for (int i = 0; i < 3; i++) {
        QCheckBox* box = new QCheckBox(options[i]);
        box->setChecked(opt_defaults[i]);
        connect(box, QOverload<bool>::of(&QCheckBox::toggled), this,
                [this, key = std::string(opt_keys[i])](bool checked) { onOptionToggled(QString::fromStdString(key), checked); });
        option_checkboxes[opt_keys[i]] = box;
        controls_layout->addWidget(box);
    }

    // Help
    controls_layout->addSpacing(12);
    help_label = new QLabel("Keys: 1-4=Mode  p=Cycle\nL=Labels  V=Video2\n[/]=FOV  q=Quit");
    help_label->setStyleSheet("QLabel { color: #98a7d9; font-size: 10px; line-height: 1.4; }");
    help_label->setWordWrap(true);
    controls_layout->addWidget(help_label);

    controls_layout->addStretch();

    // Assemble layout
    body_row->addWidget(left_label, 3);
    body_row->addWidget(right_stack, 2);
    body_row->addWidget(controls, 0);
    root_layout->addLayout(body_row);

    syncCheckboxes();
}

void DashboardWindow::onProjectionToggled(int mode, bool checked) {
    if (checked) {
        state.mode = static_cast<ProjectionMode>(mode);
    }
    syncCheckboxes();
}

void DashboardWindow::onOptionToggled(const QString& key, bool checked) {
    if (key == "labels") state.show_labels = checked;
    else if (key == "video2") state.show_secondary = checked;
    else if (key == "grid") state.show_grid = checked;
    syncCheckboxes();
}

void DashboardWindow::syncCheckboxes() {
    for (auto it = projection_checkboxes.begin(); it != projection_checkboxes.end(); ++it) {
        QCheckBox* box = it.value();
        box->blockSignals(true);
        box->setChecked(state.mode == static_cast<ProjectionMode>(it.key()));
        box->blockSignals(false);
    }
    option_checkboxes["labels"]->setChecked(state.show_labels);
    option_checkboxes["video2"]->setChecked(state.show_secondary);
    option_checkboxes["grid"]->setChecked(state.show_grid);
}

QPixmap DashboardWindow::frameToPixmap(const cv::Mat& frame) {
    cv::Mat display = frame.clone();
    if (display.empty()) {
        display = cv::Mat(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::putText(display, "NO SIGNAL", cv::Point(28, 120), cv::FONT_HERSHEY_SIMPLEX,
                    0.9, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

    if (display.channels() == 1) {
        cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    }

    cv::Mat rgb;
    cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);

    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img);
}

cv::Mat DashboardWindow::addGridOverlay(const cv::Mat& frame, int step) {
    cv::Mat result = frame.clone();
    for (int x = 0; x < result.cols; x += step) {
        cv::line(result, cv::Point(x, 0), cv::Point(x, result.rows), cv::Scalar(64, 64, 64), 1, cv::LINE_AA);
    }
    for (int y = 0; y < result.rows; y += step) {
        cv::line(result, cv::Point(0, y), cv::Point(result.cols, y), cv::Scalar(64, 64, 64), 1, cv::LINE_AA);
    }
    return result;
}

void DashboardWindow::updateDashboard(
    const cv::Mat& fisheye,
    const cv::Mat& rectified,
    const cv::Mat& secondary,
    float fps,
    const std::string& model
) {
    cv::Mat a = fisheye.clone();
    cv::Mat b = rectified.clone();
    cv::Mat c = secondary.clone();

    if (state.show_grid) {
        a = addGridOverlay(a);
        b = addGridOverlay(b);
        c = addGridOverlay(c);
    }

    QPixmap left_pix = frameToPixmap(a);
    left_label->setPixmap(left_pix.scaled(left_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QPixmap top_pix = frameToPixmap(b);
    top_label->setPixmap(top_pix.scaled(top_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QPixmap bottom_pix = frameToPixmap(c);
    bottom_label->setPixmap(bottom_pix.scaled(bottom_label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    QString status = QString::fromStdString(
        "Mode: " + std::string(state.mode == ProjectionMode::RECTILINEAR ? "RECTILINEAR" :
                               state.mode == ProjectionMode::CYLINDRICAL ? "CYLINDRICAL" :
                               state.mode == ProjectionMode::EQUIRECTANGULAR ? "EQUIRECTANGULAR" : "MERCATOR") +
        " | FPS: " + std::to_string(static_cast<int>(fps)) +
        " | Labels: " + std::string(state.show_labels ? "ON" : "OFF") +
        " | Video2: " + std::string(state.show_secondary ? "ON" : "OFF")
    );
    header_status->setText(status);
}

void DashboardWindow::closeEvent(QCloseEvent* event) {
    quit_requested = true;
    event->accept();
}

// ─── Main Entry Point ──────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Parse arguments (simplified)
    std::string device = "/dev/video0";
    std::string secondary_device = "/dev/video2";
    int width = 1280, height = 720, fps = 30;
    float lens_fov = 180.0f;

    // Open cameras
    CameraCapture cap(device, width, height, fps, "MJPG");
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera at " << device << std::endl;
        return 1;
    }

    CameraCapture sec_cap(secondary_device, width, height, fps, "MJPG");
    if (!sec_cap.isOpened()) {
        std::cerr << "[WARN] Cannot open secondary camera at " << secondary_device << std::endl;
    }

    std::cout << "[OK] " << device << ": " << cap.getWidth() << "x" << cap.getHeight()
              << " @ " << cap.getFPS() << " fps" << std::endl;

    // Estimated equidistant fisheye model: f = r_max / theta_max, centre at image centre
    int cap_w = cap.getWidth();
    int cap_h = cap.getHeight();
    float f_est = (std::min(cap_w, cap_h) / 2.0f) / (lens_fov * static_cast<float>(CV_PI) / 180.0f / 2.0f);
    cv::Mat K = (cv::Mat_<float>(3, 3) <<
                 f_est, 0, cap_w / 2.0f,
                 0, f_est, cap_h / 2.0f,
                 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_32F);

    auto unwrapper = std::make_unique<FisheyeUnwrapper>(
        cv::Size(cap_w, cap_h), K, D,
        ProjectionMode::RECTILINEAR, cv::Size(),
        110.0f, "full", lens_fov
    );

    // Create UI
    DashboardWindow window;
    window.show();

    FPSMeter meter;

    // Main loop
    cv::Mat raw, secondary, rectified;
    while (!window.shouldQuit()) {
        if (!cap.read(raw)) {
            cv::waitKey(10);
            continue;
        }

        if (sec_cap.isOpened()) {
            sec_cap.read(secondary);
        }

        // Update projection if changed
        if (unwrapper->getProjection() != window.getUIState().mode) {
            unwrapper->setProjection(window.getUIState().mode);
        }

        rectified = unwrapper->rectify(raw);
        float fps_val = meter.tick();

        if (window.getUIState().show_secondary) {
            window.updateDashboard(raw, rectified, secondary, fps_val, "equidistant");
        } else {
            cv::Mat empty = cv::Mat::zeros(secondary.size(), secondary.type());
            window.updateDashboard(raw, rectified, empty, fps_val, "equidistant");
        }

        QApplication::processEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    return 0;
}
