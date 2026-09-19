// =============================================================================
// AVM System – Main Entry Point
// =============================================================================
// Shipboard Around View Monitoring for NVIDIA Jetson Orin: EO fisheye + 2 IR
// PiP, viewpoint presets / top-view, stabilisation, AI detection, AR overlay,
// recording and logging (see docs/spec_compliance.md).
//
// Usage:
//   avm_system [config.json] [--no-display] [--duration <sec>] [--sim-motion]
//              [--tour <dir>]    demo: cycle the 3 presets (+ stabilisation), save a PNG of each, exit
//
// Keys (display window):
//   1 2 3  view presets (forward / top-view / ROI zoom)      4  raw fisheye image as captured
//   i o    show/hide IR PiP window 0 / 1        s  stabilisation on/off
//   r      recording start/stop                 q, Esc  quit
// =============================================================================

#include "core/pipeline.hpp"
#include "include/avm/avm_config.hpp"
#include "core/types.hpp"
#include "record/recording_controller.hpp"
#include "util/clock.hpp"
#include "util/logger.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef AVM_WITH_OPENCV
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "record/opencv_video_sink.hpp"
#endif

static std::atomic<bool> g_running{true};

extern "C" void signal_handler(int) { g_running.store(false); }

namespace {

struct Options {
    std::string config_path;
    bool display{true};
    double duration_s{0.0};   // 0 = run until quit
    bool sim_motion{false};
    std::string tour_dir;     // non-empty: automatic demo tour that saves screenshots
};

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-display") o.display = false;
        else if (a == "--sim-motion") o.sim_motion = true;
        else if (a == "--duration" && i + 1 < argc) o.duration_s = std::atof(argv[++i]);
        else if (a == "--tour" && i + 1 < argc) o.tour_dir = argv[++i];
        else if (a == "-h" || a == "--help") return false;
        else if (!a.empty() && a[0] != '-') o.config_path = a;
        else { std::cerr << "unknown option: " << a << "\n"; return false; }
    }
    return true;
}

// ─── Simulated sensor data (until NMEA/IMU drivers are integrated) ──────────

avm::IMUData simulate_imu(double t_s, bool motion) {
    avm::IMUData imu{};
    if (motion) {   // gentle roll/pitch to exercise the stabilisation (FR-5.1)
        imu.angles.roll = static_cast<float>(4.0 * std::sin(2.0 * M_PI * t_s / 6.0));
        imu.angles.pitch = static_cast<float>(2.0 * std::sin(2.0 * M_PI * t_s / 4.0));
    }
    return imu;
}

avm::GPSData simulate_gps(uint64_t ts) {
    avm::GPSData gps{};
    gps.latitude = 35.0890;   // Busan port area
    gps.longitude = 129.0400;
    gps.speed_knots = 8.5;
    gps.heading_true = 120.0;
    gps.timestamp_us = ts;
    return gps;
}

std::vector<avm::AISTarget> simulate_ais() {
    avm::AISTarget t1{};
    t1.mmsi = 440123000; t1.latitude = 35.0920; t1.longitude = 129.0450;
    t1.sog = 12.0f; t1.cog = 45.0f; t1.heading = 45.0f; t1.ship_type = 71;
    std::memcpy(t1.ship_name, "CONTAINER ACE", 14);
    avm::AISTarget t2{};
    t2.mmsi = 440456000; t2.latitude = 35.0860; t2.longitude = 129.0350;
    t2.sog = 4.5f; t2.cog = 270.0f; t2.heading = 275.0f; t2.ship_type = 52;
    std::memcpy(t2.ship_name, "TUG BOAT 7", 11);
    return {t1, t2};
}

#ifdef AVM_WITH_OPENCV
// Class-name labels next to the GPU-drawn boxes; only valid on the forward view (preset 0).
void draw_detection_labels(cv::Mat& img, const avm::Pipeline& pipeline) {
    avm::DetectionResult det;
    std::vector<std::string> names;
    if (!pipeline.detections(det, names)) return;
    for (const auto& b : det.boxes) {
        const std::string name = (b.class_id >= 0 && b.class_id < static_cast<int>(names.size()))
                                     ? names[static_cast<std::size_t>(b.class_id)] : "?";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s %.0f%%", name.c_str(), b.confidence * 100.0f);
        const int x = static_cast<int>(b.x * img.cols);
        const int y = static_cast<int>(b.y * img.rows);
        cv::putText(img, buf, {x + 4, std::max(24, y - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
    }
}
#endif

std::string status_line(const avm::Pipeline::Status& s, bool recording) {
    char buf[256];
    std::string ir;
    for (std::size_t i = 0; i < s.ir_connected.size(); ++i)
        ir += " IR" + std::to_string(i) + (s.ir_connected[i] ? ":ok" : ":OFF");
    std::snprintf(buf, sizeof(buf), "EO:%s%s | %.1f fps | compute mean %.1f p99 %.1f max %.1f ms | view %d%s%s",
                  s.eo_connected ? "ok" : "OFF", ir.c_str(), s.fps, s.compute_ms_mean,
                  s.compute_ms_p99, s.compute_ms_max, s.preset,
                  s.stabilization ? " STAB" : "", recording ? " REC" : "");
    return buf;
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifndef _WIN32
    std::signal(SIGQUIT, signal_handler);
#endif

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        std::cerr << "usage: avm_system [config.json] [--no-display] [--duration <sec>] [--sim-motion]\n";
        return EXIT_FAILURE;
    }

#ifdef AVM_WITH_OPENCV
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);   // our own logger reports camera problems
#endif

    // ─── Configuration (file -> env overrides -> validation) ────────────
    avm::config::AppConfig app = avm::config::default_app_config();
    if (!opt.config_path.empty()) {
        std::string err;
        if (!avm::config::load_app_config_file(opt.config_path, app, &err)) {
            std::cerr << "[FATAL] config: " << err << std::endl;
            return EXIT_FAILURE;
        }
    }
    avm::apply_env_overrides(app);
    const auto problems = avm::config::validate(app);
    if (!problems.empty()) {
        for (const auto& p : problems) std::cerr << "[FATAL] config: " << p << std::endl;
        return EXIT_FAILURE;
    }

    // ─── Logging (FR-9.2) ────────────────────────────────────────────────
    auto& log = avm::util::Logger::instance();
    log.set_console(app.logging.console);
    log.set_level(app.logging.level == "debug" ? avm::util::LogLevel::Debug
                : app.logging.level == "warn"  ? avm::util::LogLevel::Warn
                : app.logging.level == "error" ? avm::util::LogLevel::Error
                                               : avm::util::LogLevel::Info);
    if (!log.open_directory(app.logging.directory)) {
        std::cerr << "[WARN] cannot open log directory '" << app.logging.directory
                  << "'; logging to console only" << std::endl;
    }
    AVM_LOGI("main", "Shipboard AVM System (ship_avm01_ai) v0.3.0 | config: %s",
             opt.config_path.empty() ? "(defaults)" : opt.config_path.c_str());
    AVM_LOGI("main", "EO %ux%u, IR x%zu, output %ux%u, budget %.0f ms",
             app.eo.width, app.eo.height, app.ir.size(), app.output.width, app.output.height,
             static_cast<double>(avm::constants::FRAME_BUDGET_MS));

    // ─── Pipeline ────────────────────────────────────────────────────────
    avm::Pipeline pipeline(avm::make_pipeline_config(app));
    if (!pipeline.initialize()) {
        AVM_LOGE("main", "pipeline initialization failed");
        return EXIT_FAILURE;
    }

    // ─── Recording (FR-9.1) and display buffer, fed from the pipeline thread ──
    std::unique_ptr<avm::record::IVideoSink> sink;
#ifdef AVM_WITH_OPENCV
    sink = std::make_unique<avm::record::OpenCvVideoSink>();
#endif
    avm::record::RecordingController recorder(app.recording, std::move(sink));

    std::mutex frame_mu;
#ifdef AVM_WITH_OPENCV
    cv::Mat display_frame;   // RGBA copy for the UI thread
#endif
    const bool want_display = opt.display;
    const bool want_frames = opt.display || !opt.tour_dir.empty();
    pipeline.setOutputCallback([&](const avm::CameraFrame& f) {
        if (recorder.is_recording()) {
            recorder.on_frame(f.host_data, f.host_pitch, 4);
        }
#ifdef AVM_WITH_OPENCV
        if (want_frames) {
            cv::Mat view(static_cast<int>(f.spec.height), static_cast<int>(f.spec.width), CV_8UC4,
                         f.host_data, f.host_pitch);
            std::lock_guard<std::mutex> lk(frame_mu);
            view.copyTo(display_frame);
        }
#else
        (void)f; (void)want_frames;
#endif
    });

    if (!pipeline.start()) {
        AVM_LOGE("main", "pipeline start failed");
        return EXIT_FAILURE;
    }

    // ─── Sensor feed thread (simulated) ──────────────────────────────────
    std::thread sensors([&] {
        const double t0 = avm::util::steady_now_ms();
        const auto ais = simulate_ais();
        while (g_running.load()) {
            const double t = (avm::util::steady_now_ms() - t0) / 1000.0;
            const uint64_t ts = avm::util::steady_now_us();
            pipeline.updateIMU(simulate_imu(t, opt.sim_motion));
            pipeline.updateGPS(simulate_gps(ts), 120.0f);
            pipeline.updateAISTargets(ais);
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    // ─── UI / status loop ────────────────────────────────────────────────
    const auto t_start = std::chrono::steady_clock::now();
    auto last_print = t_start;
    [[maybe_unused]] bool display_ok = false;

#ifdef AVM_WITH_OPENCV
    if (want_display) {
        try {
            cv::namedWindow("AVM", cv::WINDOW_NORMAL);
            display_ok = true;
        } catch (const cv::Exception& e) {
            AVM_LOGW("main", "display unavailable (%s); running headless", e.what());
        }
    }
#endif

    [[maybe_unused]] auto toggle_recording = [&] {
        if (recorder.is_recording()) {
            recorder.stop();
            AVM_LOGI("main", "recording stopped (%llu frames): %s",
                     static_cast<unsigned long long>(recorder.frames_written()), recorder.current_path().c_str());
        } else if (recorder.start(static_cast<int>(app.output.width), static_cast<int>(app.output.height),
                                  avm::record::local_timestamp_string())) {
            AVM_LOGI("main", "recording started: %s", recorder.current_path().c_str());
        } else {
            AVM_LOGE("main", "cannot start recording: %s", recorder.last_error().c_str());
        }
    };

    // ─── automatic demo tour: preset 0 -> 1 -> 2 -> stabilisation, one PNG each ──
    struct TourStep { double at_s; int preset; bool stab; const char* file; };
    const std::vector<TourStep> tour = {
        {3.0, 3, false, "0_raw_fisheye.png"},
        {6.5, 0, false, "1_forward.png"},
        {10.0, 1, false, "2_topview.png"},
        {13.5, 2, false, "3_roi_zoom.png"},
        {17.0, 0, true, "4_forward_stabilised.png"},
    };
    std::size_t tour_next = 0;
    bool tour_selected = false;
    if (!opt.tour_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(opt.tour_dir, ec);
        pipeline.selectPreset(3);
    }

    while (g_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (opt.duration_s > 0.0 && elapsed >= opt.duration_s) break;

#ifdef AVM_WITH_OPENCV
        if (!opt.tour_dir.empty() && tour_next < tour.size()) {
            const TourStep& st = tour[tour_next];
            if (!tour_selected && elapsed >= st.at_s - 2.0) {          // switch 2 s before the snapshot
                pipeline.selectPreset(st.preset);
                pipeline.setStabilization(st.stab);
                tour_selected = true;
            }
            if (elapsed >= st.at_s) {
                cv::Mat shot;
                {
                    std::lock_guard<std::mutex> lk(frame_mu);
                    if (!display_frame.empty()) cv::cvtColor(display_frame, shot, cv::COLOR_RGBA2BGR);
                }
                if (!shot.empty()) {
                    const auto stt = pipeline.status();
                    draw_detection_labels(shot, pipeline);
                    cv::putText(shot, status_line(stt, false), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                {0, 255, 255}, 2);
                    const std::string path = opt.tour_dir + "/" + st.file;
                    cv::imwrite(path, shot);
                    AVM_LOGI("tour", "saved %s (view %d%s)", path.c_str(), stt.preset, stt.stabilization ? ", stabilised" : "");
                }
                ++tour_next;
                tour_selected = false;
                if (tour_next >= tour.size()) break;
            }
        }
#endif

#ifdef AVM_WITH_OPENCV
        if (display_ok) {
            cv::Mat shown;
            {
                std::lock_guard<std::mutex> lk(frame_mu);
                if (!display_frame.empty()) cv::cvtColor(display_frame, shown, cv::COLOR_RGBA2BGR);
            }
            if (!shown.empty()) {
                const auto st = pipeline.status();                       // FR-8.3
                draw_detection_labels(shown, pipeline);
                cv::putText(shown, status_line(st, recorder.is_recording()), {12, 28},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 2);
                cv::imshow("AVM", shown);
            }
            const int key = cv::waitKey(15) & 0xFF;
            if (key == 'q' || key == 27) break;
            else if (key == '1') pipeline.selectPreset(0);
            else if (key == '2') pipeline.selectPreset(1);
            else if (key == '3') pipeline.selectPreset(2);
            else if (key == '4') pipeline.selectPreset(3);   // raw fisheye (as captured)
            else if (key == 'i') pipeline.togglePip(0);
            else if (key == 'o') pipeline.togglePip(1);
            else if (key == 's') pipeline.setStabilization(!pipeline.status().stabilization);
            else if (key == 'r') toggle_recording();
        } else
#endif
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (now - last_print >= std::chrono::seconds(1)) {
            AVM_LOGI("status", "%s", status_line(pipeline.status(), recorder.is_recording()).c_str());
            last_print = now;
        }
    }

    // ─── Shutdown ────────────────────────────────────────────────────────
    g_running = false;
    AVM_LOGI("main", "shutting down");
    if (sensors.joinable()) sensors.join();
    if (recorder.is_recording()) recorder.stop();
    pipeline.stop();
#ifdef AVM_WITH_OPENCV
    if (display_ok) cv::destroyAllWindows();
#endif
    AVM_LOGI("main", "shutdown complete (%llu frames, %zu errors logged)",
             static_cast<unsigned long long>(pipeline.status().frames), log.error_count());
    log.close();
    return EXIT_SUCCESS;
}
