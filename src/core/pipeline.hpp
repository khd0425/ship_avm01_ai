#pragma once

#include "src/core/types.hpp"
#include "src/core/error_checks.hpp"
#include "include/avm/avm_config.hpp"
#include "src/capture/capture_manager.hpp"
#include "src/undistort/undistort_engine.hpp"
#include "src/render/morphing_renderer.hpp"
#include "src/render/ar_overlay.hpp"
#include "src/inference/inference_engine.hpp"
#include "src/inference/dnn_detector.hpp"
#include "src/imgproc/agc.hpp"
#include "src/metrics/perf_stats.hpp"
#include "src/sync/frame_sync.hpp"
#include "src/view/pip_layout.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace avm {

// ─── Pipeline ──────────────────────────────────────────────────────────────
//
// Software flow of the requirement spec (3.4), driven by one processing thread:
//
//   capture threads (EO + IR x2, mapped pinned memory, zero-copy)
//        │ EO frame (newest)                 IR frames (newest each)
//        ▼                                        ▼
//   time sync (EO master, 16 ms)  ──────►  AGC (histogram LUT, host)
//        │                                        │
//        ├─► undistort LUT ─► AI detector (async, own stream)
//        ▼                                        │
//   view render (preset / top-view / stabilised)  │
//        ▼                                        ▼
//   IR PiP composite  ◄───────────────────────────┘
//        ▼
//   AR overlay ─► output frame (RGBA, mapped) ─► display / recording callback
//
// Performance targets: >= 30 fps, compute latency <= 33 ms (PR-1, PR-2).
//
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    bool initialize();
    bool start();
    void stop();

    /// Latest output frame (RGBA8, mapped: host_data is readable by the CPU).
    const CameraFrame* output() const { return output_frame_.get(); }

    /// Called from the pipeline thread after every completed frame.
    using OutputCallback = std::function<void(const CameraFrame&)>;
    void setOutputCallback(OutputCallback cb) { output_cb_ = std::move(cb); }

    // ─── sensor input (any thread) ──────────────────────────────────────
    void updateAISTargets(const std::vector<AISTarget>& targets);
    void updateIMU(const IMUData& imu);
    void updateGPS(const GPSData& gps, float heading);

    // ─── operator controls (UI thread) ──────────────────────────────────
    void selectPreset(int index);              // FR-4.1
    void setStabilization(bool enabled);       // FR-5.1
    void togglePip(int ir_index);              // FR-3.3
    void movePip(int ir_index, double nx, double ny);   // drag, normalised top-left
    /// Which PiP window is at a screen position (or -1).
    int pipAt(int x, int y) const;

    // ─── status (FR-8.3) ────────────────────────────────────────────────
    struct Status {
        bool eo_connected{false};
        std::vector<bool> ir_connected;
        std::vector<bool> pip_visible;
        int preset{0};
        bool stabilization{false};
        double fps{0.0};
        double compute_ms_mean{0.0};
        double compute_ms_p99{0.0};
        double compute_ms_max{0.0};
        double over_budget_fraction{0.0};   // frames above 33 ms
        std::uint64_t frames{0};
    };
    Status status() const;

    /// Per-stage timing of the last frame (ms).
    struct Stats {
        float undistort_ms{0.0f};
        float render_ms{0.0f};
        float pip_ms{0.0f};
        float overlay_ms{0.0f};
        float inference_ms{0.0f};
        float total_ms{0.0f};
        float gpu_stage_ms{0.0f};     // GPU-side (CUDA events) on the view stream
        float gpu_view_ms{0.0f};
        float gpu_pip_ms{0.0f};
        float gpu_overlay_ms{0.0f};
        std::uint32_t frame_count{0};
    };
    Stats stats() const;

    /// Latest detections (normalised, forward-view coordinates) and class names, for UI labels.
    bool detections(DetectionResult& out, std::vector<std::string>& class_names) const;

private:
    void run();
    void processFrame(const CameraFrame& eo);
    void logSyncSummary(std::uint64_t now_us);

    struct IrState {
        imgproc::AgcController agc;
        unique_cuda_ptr d_lut;                 // 4096 bytes
        explicit IrState(const imgproc::AgcParams& p) : agc(p) {}
    };

    std::unique_ptr<CameraFrame> makeDeviceFrame(uint32_t w, uint32_t h, PixelFormat fmt,
                                                 unique_cuda_ptr& storage);

    PipelineConfig config_;

    // Sub-modules
    std::unique_ptr<CaptureManager>    capture_;
    std::unique_ptr<UndistortEngine>   undistort_eo_;
    std::unique_ptr<MorphingRenderer>  renderer_;
    std::unique_ptr<AROverlay>         overlay_;
    std::unique_ptr<InferenceEngine>   inference_;    // TensorRT path (stub until the Jetson)
#ifdef AVM_WITH_OPENCV
    std::unique_ptr<DnnDetector>       dnn_;          // OpenCV DNN path (development PC)
    MappedBuffer                       dnn_host_;     // host copy of the rectified view for the detector
    DetectionResult                    latest_det_;      // last detector output, normalised to the detector input image
    DetectionResult                    latest_det_out_;  // same boxes mapped to output-frame coordinates (overlay + labels)
    int                                det_channel_{-1}; // camera channel the detector reads (-1 = rectified EO view)
    mutable std::mutex                 det_mu_;
#endif
    std::vector<std::unique_ptr<IrState>> ir_;

    // CUDA streams: [UNDISTORT] = view/PiP/overlay, [INFERENCE] = detector
    std::vector<CudaStream> streams_;
    std::unique_ptr<CudaEvent> undistort_done_;

    // Discrete GPU (desktop): mapped host memory is read over PCIe, which is far too
    // slow for the random-access view kernel, so the EO frame is staged into a
    // device buffer once per frame. On Jetson (integrated GPU) this is skipped.
    bool discrete_gpu_{false};
    unique_cuda_ptr eo_stage_storage_;
    std::unique_ptr<CameraFrame> eo_stage_;
    std::unique_ptr<CudaEvent> eo_staged_;
    std::vector<std::unique_ptr<CudaEvent>> gpu_ev_;   // GPU timeline on the view stream: start, staged, view, pip, overlay

    // Discrete GPU: render into device memory, then one D2H copy into the mapped output
    // (per-pixel writes to host memory over PCIe are ~10x slower than the kernel itself).
    unique_cuda_ptr output_dev_storage_;
    std::unique_ptr<CameraFrame> output_dev_;
    std::vector<unique_cuda_ptr> ir_stage_storage_;
    std::vector<std::unique_ptr<CameraFrame>> ir_stage_;

    // Buffers
    unique_cuda_ptr undistorted_storage_;
    std::unique_ptr<CameraFrame> undistorted_eo_;   // rectified forward view (detector input)
    MappedBuffer output_buffer_;
    std::unique_ptr<CameraFrame> output_frame_;

    // Time sync (capture threads push stamps, pipeline thread evaluates)
    mutable std::mutex sync_mu_;
    std::unique_ptr<sync::FrameSynchronizer> sync_;
    std::uint64_t last_sync_log_us_{0};

    // Operator controls
    mutable std::mutex ui_mu_;
    std::unique_ptr<view::PipLayout> pip_;

    // Latest sensor data
    std::mutex sensor_mu_;
    IMUData latest_imu_{};
    GPSData latest_gps_{};
    float latest_heading_{0.0f};
    std::vector<AISTarget> latest_targets_;

    // Metrics
    mutable std::mutex stats_mu_;
    Stats stats_;
    metrics::RollingStats compute_ms_{240};
    metrics::FpsCounter fps_{60};

    OutputCallback output_cb_{nullptr};
    std::thread thread_;
    std::atomic<bool> running_{false};
    bool initialized_{false};
};

} // namespace avm
