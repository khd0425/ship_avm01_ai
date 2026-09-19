#include "pipeline.hpp"

#include "src/render/cuda_pip.hpp"
#include "src/util/clock.hpp"
#include "src/util/logger.hpp"

#include <chrono>
#include <cstring>

namespace avm {

namespace {

CameraCalibration to_calibration(const geo::FisheyeModel& m) {
    CameraCalibration c{};
    c.type = CameraType::EO_FISHEYE;
    c.intr.fx = static_cast<float>(m.fx);
    c.intr.fy = static_cast<float>(m.fy);
    c.intr.cx = static_cast<float>(m.cx);
    c.intr.cy = static_cast<float>(m.cy);
    c.intr.k1 = static_cast<float>(m.k1);
    c.intr.k2 = static_cast<float>(m.k2);
    c.intr.k3 = static_cast<float>(m.k3);
    c.intr.p1 = c.intr.p2 = 0.0f;
    c.intr.k4 = static_cast<float>(m.k4);
    c.fov_degrees = static_cast<float>(m.fov_deg);
    return c;
}

std::size_t bytes_per_pixel(PixelFormat f) {
    return f == PixelFormat::RGBA8 ? 4 : f == PixelFormat::RGB8 ? 3 : f == PixelFormat::MONO16 ? 2 : 1;
}

} // namespace

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config) {
    for (int i = 0; i < constants::NUM_CUDA_STREAMS; ++i) {
        streams_.emplace_back();
    }
}

Pipeline::~Pipeline() {
    stop();
}

std::unique_ptr<CameraFrame> Pipeline::makeDeviceFrame(uint32_t w, uint32_t h, PixelFormat fmt,
                                                       unique_cuda_ptr& storage) {
    auto f = std::make_unique<CameraFrame>();
    f->spec = CameraSpec{CameraType::EO_FISHEYE, w, h, fmt, config_.eo_spec.fps, 0};
    const std::size_t pitch = ((w * bytes_per_pixel(fmt) + 255) / 256) * 256;
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, pitch * h));
    storage.reset(p);
    f->device_data = p;
    f->device_pitch = pitch;
    f->stream = streams_[constants::STREAM_UNDISTORT].get();
    return f;
}

bool Pipeline::initialize() {
    if (initialized_) return true;
    const config::AppConfig& app = config_.app;

    try {
        // Best effort: allow mapped host memory on devices that need the flag.
        cudaSetDeviceFlags(cudaDeviceMapHost);
        cudaGetLastError();

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        discrete_gpu_ = (prop.integrated == 0);
        AVM_LOGI("pipeline", "GPU: %s (sm_%d%d, %s) -> %s", prop.name, prop.major, prop.minor,
                 discrete_gpu_ ? "discrete, EO frames are staged to device memory"
                               : "integrated, zero-copy mapped memory",
                 discrete_gpu_ ? "PC mode" : "Jetson mode");

        // 1. Capture (mapped pinned rings, camera sources)
        capture_ = std::make_unique<CaptureManager>(config_);
        if (!capture_->initialize()) {
            AVM_LOGE("pipeline", "CaptureManager init failed");
            return false;
        }

        // 2. Time sync: only the newest IR frame of each channel is reachable
        //    in the capture ring, so one stamp of history is kept.
        sync::SyncConfig sc = app.sync;
        sc.history = 1;
        sync_ = std::make_unique<sync::FrameSynchronizer>(capture_->num_ir_cameras(), sc);

        // 3. Rectified forward view (LUT) for the AI detector.
        const geo::CameraPose pose = geo::make_camera_pose(app.mount.height_m, app.mount.pitch_deg,
                                                           app.mount.roll_deg);
        geo::ViewParams forward;
        forward.hfov_deg = app.view.presets.forward_hfov_deg;
        forward.pitch_deg = app.view.presets.forward_pitch_deg;
        undistort_eo_ = std::make_unique<UndistortEngine>(
            app.eo_intrinsics, pose, forward, config_.output_width, config_.output_height,
            config_.eo_spec.width, config_.eo_spec.height);

        // 4. Renderer (presets, top-view, stabilisation)
        renderer_ = std::make_unique<MorphingRenderer>(config_.output_width, config_.output_height, app);

        // 5. AR overlay
        overlay_ = std::make_unique<AROverlay>(config_.output_width, config_.output_height,
                                               to_calibration(app.eo_intrinsics));

        // 6. Detector
        if (config_.enable_inference) {
#ifdef AVM_WITH_OPENCV
            if (app.inference.backend == "opencv_dnn") {
                dnn_ = std::make_unique<DnnDetector>();
                DnnDetector::Config dc;
                dc.model_path = app.inference.model_path;
                dc.input_size = app.inference.input_size;
                dc.confidence = static_cast<float>(app.inference.confidence_threshold);
                dc.nms = static_cast<float>(app.inference.nms_threshold);
                dc.classes = app.inference.classes;
                dc.arch = app.inference.arch;
                if (app.inference.source == "ir0") det_channel_ = 0;
                else if (app.inference.source == "ir1") det_channel_ = 1;
                std::string derr;
                if (!dnn_->start(dc, &derr)) {
                    AVM_LOGE("pipeline", "detector unavailable: %s", derr.c_str());
                    dnn_.reset();
                }
            } else
#endif
            if (app.inference.backend != "opencv_dnn") {
                inference_ = std::make_unique<InferenceEngine>(config_);
            }
        }

        // 7. IR: AGC state + device LUT
        for (int i = 0; i < capture_->num_ir_cameras(); ++i) {
            auto st = std::make_unique<IrState>(app.agc);
            void* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, 4096));
            st->d_lut.reset(p);
            ir_.push_back(std::move(st));
        }
        pip_ = std::make_unique<view::PipLayout>(
            app.pip.windows,
            app.ir.empty() ? 4.0 / 3.0 : static_cast<double>(app.ir[0].width) / app.ir[0].height);

        // 8. Buffers
        undistorted_eo_ = makeDeviceFrame(config_.output_width, config_.output_height,
                                          PixelFormat::RGB8, undistorted_storage_);
        const std::size_t out_pitch = ((config_.output_width * 4 + 255) / 256) * 256;
        output_buffer_ = alloc_mapped(out_pitch * config_.output_height);
        output_frame_ = std::make_unique<CameraFrame>();
        output_frame_->spec = CameraSpec{CameraType::EO_FISHEYE, config_.output_width, config_.output_height,
                                         PixelFormat::RGBA8, config_.eo_spec.fps, 0};
        output_frame_->host_data = output_buffer_.host.get();
        output_frame_->host_pitch = out_pitch;
        output_frame_->device_data = output_buffer_.device;
        output_frame_->device_pitch = out_pitch;
        CUDA_CHECK(cudaMemset(output_buffer_.device, 0, out_pitch * config_.output_height));

        undistort_done_ = std::make_unique<CudaEvent>();
#ifdef AVM_WITH_OPENCV
        if (dnn_) dnn_host_ = alloc_mapped(undistorted_eo_->device_pitch * config_.output_height);
#endif
        for (int i = 0; i < 5; ++i) gpu_ev_.push_back(std::make_unique<CudaEvent>());
        if (discrete_gpu_) {
            eo_stage_ = makeDeviceFrame(config_.eo_spec.width, config_.eo_spec.height,
                                        PixelFormat::RGB8, eo_stage_storage_);
            eo_stage_->spec = config_.eo_spec;
            eo_staged_ = std::make_unique<CudaEvent>();
            output_dev_ = makeDeviceFrame(config_.output_width, config_.output_height,
                                          PixelFormat::RGBA8, output_dev_storage_);
            for (int i = 0; i < capture_->num_ir_cameras(); ++i) {
                ir_stage_storage_.emplace_back();
                ir_stage_.push_back(makeDeviceFrame(config_.ir_specs[static_cast<std::size_t>(i)].width,
                                                    config_.ir_specs[static_cast<std::size_t>(i)].height,
                                                    config_.ir_specs[static_cast<std::size_t>(i)].pixel_format,
                                                    ir_stage_storage_.back()));
            }
        }

        // 9. LUTs / engines
        cudaStream_t s = streams_[constants::STREAM_UNDISTORT].get();
        if (!undistort_eo_->buildLUTs(s)) {
            AVM_LOGE("pipeline", "EO LUT build failed");
            return false;
        }
        if (inference_ && !inference_->loadEngine(s)) {
            AVM_LOGE("pipeline", "inference engine load failed");
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
    } catch (const std::exception& e) {
        AVM_LOGE("pipeline", "initialization failed: %s", e.what());
        return false;
    }

    initialized_ = true;
    AVM_LOGI("pipeline", "initialization complete (%ux%u output, %d IR)", config_.output_width,
             config_.output_height, capture_->num_ir_cameras());
    return true;
}

bool Pipeline::start() {
    if (!initialized_) return false;
    if (running_) return true;

    running_ = true;
    const bool ok = capture_->start([this](int channel, const sync::FrameStamp& st, std::uint64_t now_us) {
        std::lock_guard<std::mutex> lk(sync_mu_);
        if (channel < 0) sync_->push_eo(st, now_us);
        else sync_->push_ir(channel, st, now_us);
    });
    if (!ok) {
        running_ = false;
        return false;
    }
    thread_ = std::thread([this] { run(); });
    AVM_LOGI("pipeline", "started");
    return true;
}

void Pipeline::stop() {
    if (!running_.exchange(false)) return;
    if (capture_) capture_->stop();
    if (thread_.joinable()) thread_.join();
    for (auto& s : streams_) {
        cudaStreamSynchronize(s.get());   // best effort; no throw from stop()
    }
    AVM_LOGI("pipeline", "stopped");
}

void Pipeline::run() {
    std::uint64_t last_seq = 0;
    std::uint64_t errors = 0;
    while (running_) {
        std::uint64_t seq = 0;
        const CameraFrame* eo = capture_->acquireEo(last_seq, seq, std::chrono::milliseconds(100));
        if (!eo) continue;   // EO not delivering; status() reports eo_connected=false
        last_seq = seq;
        try {
            processFrame(*eo);
        } catch (const std::exception& e) {
            // A failed frame must never take the system down (PR-6, FR-1.4).
            if (++errors == 1 || errors % 100 == 0) {
                AVM_LOGE("pipeline", "frame %llu failed: %s (total %llu)",
                         static_cast<unsigned long long>(seq), e.what(),
                         static_cast<unsigned long long>(errors));
            }
        }
        capture_->releaseEo();
    }
}

void Pipeline::processFrame(const CameraFrame& eo) {
    using clock = std::chrono::steady_clock;
    const auto t_frame = clock::now();
    auto ms_since = [](clock::time_point a) {
        return std::chrono::duration<float, std::milli>(clock::now() - a).count();
    };

    const std::uint64_t now_us = util::steady_now_us();
    const double now_ms = static_cast<double>(now_us) / 1000.0;
    cudaStream_t s_view = streams_[constants::STREAM_UNDISTORT].get();
    cudaStream_t s_inf = streams_[constants::STREAM_INFERENCE].get();
    const int W = static_cast<int>(config_.output_width);
    const int H = static_cast<int>(config_.output_height);

    Stats st;

    // ─── 0. discrete GPU: stage the EO frame into device memory ─────────
    CameraFrame* target = discrete_gpu_ ? output_dev_.get() : output_frame_.get();
    const CameraFrame* eo_src = &eo;
    CameraFrame eo_dev;
    gpu_ev_[0]->record(s_view);
    if (discrete_gpu_) {
        CUDA_CHECK(cudaMemcpy2DAsync(eo_stage_->device_data, eo_stage_->device_pitch,
                                     eo.host_data, eo.host_pitch,
                                     static_cast<size_t>(eo.spec.width) * 3, eo.spec.height,
                                     cudaMemcpyHostToDevice, s_view));
        eo_staged_->record(s_view);
        CUDA_CHECK(cudaStreamWaitEvent(s_inf, eo_staged_->get(), 0));
        eo_dev = eo;
        eo_dev.device_data = eo_stage_->device_data;
        eo_dev.device_pitch = eo_stage_->device_pitch;
        eo_src = &eo_dev;
    }

    gpu_ev_[1]->record(s_view);

    // ─── 1. detector: fetch the PREVIOUS result, launch the new one ─────
    float t_inf = 0.0f;
    bool have_detections = false;
    DetectionResult det;
    if (inference_) {
        auto t = clock::now();
        have_detections = inference_->fetchResults(det);
        undistort_eo_->correct(*eo_src, *undistorted_eo_, s_inf);
        undistort_done_->record(s_inf);
        inference_->inferAsync(*undistorted_eo_, s_inf);
        t_inf = ms_since(t);
    }
#ifdef AVM_WITH_OPENCV
    if (dnn_) {
        auto t = clock::now();
        DetectionResult fresh;
        if (dnn_->fetch(fresh)) {
            std::lock_guard<std::mutex> lk(det_mu_);
            latest_det_ = fresh;
            det = fresh;
            have_detections = det_channel_ < 0;   // camera-channel detections are mapped in the PiP loop
            AVM_LOGD("detector", "%zu box(es), %.0f ms", fresh.boxes.size(), dnn_->last_inference_ms());
        }
        if (det_channel_ < 0 && dnn_->idle()) {   // rectify the forward view only when the detector can take it
            undistort_eo_->correct(*eo_src, *undistorted_eo_, s_inf);
            CUDA_CHECK(cudaMemcpy2DAsync(dnn_host_.device, undistorted_eo_->device_pitch,
                                         undistorted_eo_->device_data, undistorted_eo_->device_pitch,
                                         static_cast<size_t>(W) * 3, static_cast<size_t>(H),
                                         cudaMemcpyDeviceToDevice, s_inf));
            CUDA_CHECK(cudaStreamSynchronize(s_inf));
            dnn_->submit(static_cast<const std::uint8_t*>(dnn_host_.host.get()), undistorted_eo_->device_pitch, W, H);
        }
        t_inf = ms_since(t);
    }
#endif
    st.undistort_ms = t_inf;   // undistort + detector launch on the CPU side
    st.inference_ms = t_inf;

    // ─── 2. operator view ───────────────────────────────────────────────
    {
        auto t = clock::now();
        IMUData imu;
        { std::lock_guard<std::mutex> lk(sensor_mu_); imu = latest_imu_; }
        renderer_->updateAttitude(imu);
        renderer_->render(*eo_src, *target, s_view, now_ms);
        st.render_ms = ms_since(t);
    }
    gpu_ev_[2]->record(s_view);

    // ─── 3. IR PiP with AGC (FR-3.1 / FR-3.2 / FR-1.3 / FR-1.4) ─────────
    {
        auto t = clock::now();
        const int n_ir = capture_->num_ir_cameras();
        std::vector<sync::ChannelMatch> frame_matches(static_cast<std::size_t>(n_ir));
        for (int i = 0; i < n_ir; ++i) {
            std::uint64_t ir_seq = 0;
            const CameraFrame* ir = capture_->acquireIrNearest(i, eo.timestamp_us, ir_seq);

            sync::ChannelMatch m;
            {
                std::lock_guard<std::mutex> lk(sync_mu_);
                const sync::FrameStamp stamp{ir ? ir->timestamp_us : 0, ir ? ir->frame_id : 0};
                m = sync_->evaluate(i, ir ? &stamp : nullptr, eo.timestamp_us, now_us);
            }
            frame_matches[static_cast<std::size_t>(i)] = m;

            bool visible;
            render::PipRect rect{};
            {
                std::lock_guard<std::mutex> lk(ui_mu_);
                visible = config_.enable_ir_pip && pip_->visible(static_cast<std::size_t>(i));
                const view::Rect r = pip_->rect(static_cast<std::size_t>(i), W, H);
                rect = {r.x, r.y, r.w, r.h};
            }

            if (visible) {
                std::uint8_t* dst = static_cast<std::uint8_t*>(target->device_data);
                const int dst_pitch = static_cast<int>(target->device_pitch);
                if (m.state == sync::ChannelState::Disconnected) {
                    render::launch_pip_inactive(dst, dst_pitch, W, H, rect, s_view);        // FR-1.4
                } else if (m.used && ir && ir->spec.pixel_format == PixelFormat::RGB8) {
                    // colour camera in this slot (normal FOV): no AGC, show as is
                    CameraFrame ir_dev = *ir;
                    if (discrete_gpu_) {
                        CameraFrame& stage = *ir_stage_[static_cast<std::size_t>(i)];
                        CUDA_CHECK(cudaMemcpy2DAsync(stage.device_data, stage.device_pitch, ir->host_data,
                                                     ir->host_pitch, static_cast<size_t>(ir->spec.width) * 3,
                                                     ir->spec.height, cudaMemcpyHostToDevice, s_view));
                        ir_dev.device_data = stage.device_data;
                        ir_dev.device_pitch = stage.device_pitch;
                    }
                    render::launch_pip_rgb8(static_cast<const std::uint8_t*>(ir_dev.device_data),
                                            static_cast<int>(ir_dev.device_pitch),
                                            static_cast<int>(ir->spec.width), static_cast<int>(ir->spec.height),
                                            dst, dst_pitch, W, H, rect, s_view);
                    CUDA_CHECK(cudaStreamSynchronize(s_view));
                } else if (m.used && ir) {                                                   // FR-1.3
                    IrState& irs = *ir_[static_cast<std::size_t>(i)];
                    CameraFrame ir_dev = *ir;                    // device-side view of this IR frame
                    if (discrete_gpu_) {
                        CameraFrame& stage = *ir_stage_[static_cast<std::size_t>(i)];
                        const std::size_t row = static_cast<std::size_t>(ir->spec.width) *
                                                (ir->spec.pixel_format == PixelFormat::MONO16 ? 2 : 1);
                        CUDA_CHECK(cudaMemcpy2DAsync(stage.device_data, stage.device_pitch, ir->host_data,
                                                     ir->host_pitch, row, ir->spec.height,
                                                     cudaMemcpyHostToDevice, s_view));
                        ir_dev.device_data = stage.device_data;
                        ir_dev.device_pitch = stage.device_pitch;
                    }
                    const int w = static_cast<int>(ir->spec.width), h = static_cast<int>(ir->spec.height);
                    if (ir->spec.pixel_format == PixelFormat::MONO16) {
                        const auto& lut = irs.agc.update16(static_cast<const std::uint16_t*>(ir->host_data),
                                                           w, h, ir->host_pitch);
                        CUDA_CHECK(cudaMemcpyAsync(irs.d_lut.get(), lut.data(), lut.size(),
                                                   cudaMemcpyHostToDevice, s_view));
                        render::launch_pip_mono16(static_cast<const std::uint16_t*>(ir_dev.device_data),
                                                  static_cast<int>(ir_dev.device_pitch), w, h,
                                                  static_cast<const std::uint8_t*>(irs.d_lut.get()),
                                                  dst, dst_pitch, W, H, rect, s_view);
                    } else {
                        const auto& lut = irs.agc.update8(static_cast<const std::uint8_t*>(ir->host_data),
                                                          w, h, ir->host_pitch);
                        CUDA_CHECK(cudaMemcpyAsync(irs.d_lut.get(), lut.data(), lut.size(),
                                                   cudaMemcpyHostToDevice, s_view));
                        render::launch_pip_mono8(static_cast<const std::uint8_t*>(ir_dev.device_data),
                                                 static_cast<int>(ir_dev.device_pitch), w, h,
                                                 static_cast<const std::uint8_t*>(irs.d_lut.get()),
                                                 dst, dst_pitch, W, H, rect, s_view);
                    }
                    // The AGC LUT copy above is pageable-host -> device, so it has
                    // been staged before the call returns; the IR buffer itself must
                    // stay valid until the kernel is done.
                    CUDA_CHECK(cudaStreamSynchronize(s_view));
                }
                // out of tolerance: window not drawn for this frame (spec FR-1.3)
            }
#ifdef AVM_WITH_OPENCV
            if (dnn_ && det_channel_ == i) {
                // The detector works on this camera's own image (normal FOV, no geometry needed).
                if (ir && m.state == sync::ChannelState::Active && ir->spec.pixel_format == PixelFormat::RGB8 && dnn_->idle()) {
                    dnn_->submit(static_cast<const std::uint8_t*>(ir->host_data), ir->host_pitch,
                                 static_cast<int>(ir->spec.width), static_cast<int>(ir->spec.height));
                }
                // Map boxes (normalised to the camera image) into the PiP window on the output frame.
                DetectionResult mapped;
                if (visible && m.state == sync::ChannelState::Active) {
                    std::lock_guard<std::mutex> lk(det_mu_);
                    mapped = latest_det_;
                    for (auto& b : mapped.boxes) {
                        b.x = (rect.x + b.x * rect.w) / static_cast<float>(W);
                        b.y = (rect.y + b.y * rect.h) / static_cast<float>(H);
                        b.w = b.w * rect.w / static_cast<float>(W);
                        b.h = b.h * rect.h / static_cast<float>(H);
                    }
                }
                det = mapped;
                have_detections = true;
            }
#endif
            if (ir) capture_->releaseIr(i);
        }
        st.pip_ms = ms_since(t);
        gpu_ev_[3]->record(s_view);
        AVM_LOGD("sync", "%s", sync::FrameSynchronizer::format_match_log(eo.timestamp_us, frame_matches).c_str());
    }

    // ─── 4. AR overlay (optional FR-6) ──────────────────────────────────
    if (config_.enable_ar_overlay) {
        auto t = clock::now();
        {
            std::lock_guard<std::mutex> lk(sensor_mu_);
            overlay_->updateNavigation(latest_gps_, latest_heading_);
            overlay_->updateTargets(latest_targets_);
        }
        // Detection boxes are in forward-view coordinates: only drawn there.
        // EO-view detections are in forward-view coordinates: only valid on that preset.
        const bool boxes_valid = (det_channel_ >= 0) || renderer_->detectionOverlayValid(now_ms);
        if (have_detections) {
            DetectionResult shown = boxes_valid ? det : DetectionResult{};
            overlay_->updateDetections(shown);
#ifdef AVM_WITH_OPENCV
            std::lock_guard<std::mutex> lk(det_mu_);
            latest_det_out_ = shown;
#endif
        }
        overlay_->render(*target, s_view);
        st.overlay_ms = ms_since(t);
    }

    gpu_ev_[4]->record(s_view);
    if (discrete_gpu_) {   // publish the finished frame to the (host-visible) output buffer
        CUDA_CHECK(cudaMemcpy2DAsync(output_frame_->device_data, output_frame_->device_pitch,
                                     output_dev_->device_data, output_dev_->device_pitch,
                                     static_cast<size_t>(W) * 4, static_cast<size_t>(H),
                                     cudaMemcpyDeviceToDevice, s_view));
    }
    CUDA_CHECK(cudaStreamSynchronize(s_view));
    undistort_done_->sync();
    st.gpu_stage_ms = gpu_ev_[1]->elapsed_since(*gpu_ev_[0]);
    st.gpu_view_ms = gpu_ev_[2]->elapsed_since(*gpu_ev_[1]);
    st.gpu_pip_ms = gpu_ev_[3]->elapsed_since(*gpu_ev_[2]);
    st.gpu_overlay_ms = gpu_ev_[4]->elapsed_since(*gpu_ev_[3]);   // the raw EO frame may be recycled after this

    st.total_ms = ms_since(t_frame);
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.undistort_ms = st.undistort_ms;
        stats_.render_ms = st.render_ms;
        stats_.pip_ms = st.pip_ms;
        stats_.overlay_ms = st.overlay_ms;
        stats_.inference_ms = st.inference_ms;
        stats_.total_ms = st.total_ms;
        stats_.gpu_stage_ms = st.gpu_stage_ms;
        stats_.gpu_view_ms = st.gpu_view_ms;
        stats_.gpu_pip_ms = st.gpu_pip_ms;
        stats_.gpu_overlay_ms = st.gpu_overlay_ms;
        ++stats_.frame_count;
        compute_ms_.add(st.total_ms);
        fps_.tick(now_ms / 1000.0);
    }

    logSyncSummary(now_us);

    if (output_cb_) output_cb_(*output_frame_);
}

void Pipeline::logSyncSummary(std::uint64_t now_us) {
    if (now_us - last_sync_log_us_ < 1'000'000) return;
    last_sync_log_us_ = now_us;
    {
        Stats s;
        { std::lock_guard<std::mutex> lk(stats_mu_); s = stats_; }
        AVM_LOGD("perf", "last frame: total %.1f ms | CPU: detector %.1f view-launch %.1f pip %.1f overlay %.1f | "
                 "GPU: EO-upload %.1f view %.1f pip %.1f overlay %.1f",
                 s.total_ms, s.undistort_ms, s.render_ms, s.pip_ms, s.overlay_ms,
                 s.gpu_stage_ms, s.gpu_view_ms, s.gpu_pip_ms, s.gpu_overlay_ms);
    }
    std::lock_guard<std::mutex> lk(sync_mu_);
    for (int i = 0; i < sync_->num_ir_channels(); ++i) {
        const sync::ChannelStats& s = sync_->stats(i);
        const bool up = sync_->state(i, now_us) == sync::ChannelState::Active;
        AVM_LOGI("sync", "IR%d %s matched=%llu out_of_tol=%llu dropped=%llu mean|dt|=%.2fms max|dt|=%.2fms",
                 i, up ? "ACTIVE" : "DISCONNECTED",
                 static_cast<unsigned long long>(s.matched),
                 static_cast<unsigned long long>(s.excluded_out_of_tolerance),
                 static_cast<unsigned long long>(s.excluded_disconnected),
                 s.mean_abs_delta_us / 1000.0, static_cast<double>(s.max_abs_delta_us) / 1000.0);
    }
}

// ─── operator controls ─────────────────────────────────────────────────────

void Pipeline::selectPreset(int index) { if (renderer_) renderer_->requestPreset(index); }
void Pipeline::setStabilization(bool enabled) { if (renderer_) renderer_->setStabilization(enabled); }

void Pipeline::togglePip(int ir_index) {
    std::lock_guard<std::mutex> lk(ui_mu_);
    if (pip_ && ir_index >= 0) pip_->toggle(static_cast<std::size_t>(ir_index));
}

void Pipeline::movePip(int ir_index, double nx, double ny) {
    std::lock_guard<std::mutex> lk(ui_mu_);
    if (pip_ && ir_index >= 0)
        pip_->move_to(static_cast<std::size_t>(ir_index), nx, ny,
                      static_cast<int>(config_.output_width), static_cast<int>(config_.output_height));
}

int Pipeline::pipAt(int x, int y) const {
    std::lock_guard<std::mutex> lk(ui_mu_);
    return pip_ ? pip_->hit_test(x, y, static_cast<int>(config_.output_width),
                                 static_cast<int>(config_.output_height)) : -1;
}

// ─── status ────────────────────────────────────────────────────────────────

Pipeline::Status Pipeline::status() const {
    Status s;
    if (!initialized_) return s;
    const std::uint64_t now_us = util::steady_now_us();
    {
        // capture-side sync state
        std::lock_guard<std::mutex> lk(sync_mu_);
        s.eo_connected = sync_->eo_connected(now_us);
        for (int i = 0; i < sync_->num_ir_channels(); ++i)
            s.ir_connected.push_back(sync_->state(i, now_us) == sync::ChannelState::Active);
    }
    {
        std::lock_guard<std::mutex> lk(ui_mu_);
        for (std::size_t i = 0; i < pip_->count(); ++i) s.pip_visible.push_back(pip_->visible(i));
    }
    s.preset = renderer_->activePreset();
    s.stabilization = renderer_->stabilization();
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        s.fps = fps_.fps();
        s.compute_ms_mean = compute_ms_.mean();
        s.compute_ms_p99 = compute_ms_.percentile(99.0);
        s.compute_ms_max = compute_ms_.max();
        s.over_budget_fraction = compute_ms_.fraction_above(constants::FRAME_BUDGET_MS);
        s.frames = stats_.frame_count;
    }
    return s;
}

bool Pipeline::detections(DetectionResult& out, std::vector<std::string>& class_names) const {
#ifdef AVM_WITH_OPENCV
    if (dnn_) {
        std::lock_guard<std::mutex> lk(det_mu_);
        out = latest_det_out_;
        class_names = dnn_->classes();
        return true;
    }
#endif
    (void)out; (void)class_names;
    return false;
}

Pipeline::Stats Pipeline::stats() const {
    std::lock_guard<std::mutex> lk(stats_mu_);
    return stats_;
}

// ─── sensor input ──────────────────────────────────────────────────────────

void Pipeline::updateAISTargets(const std::vector<AISTarget>& targets) {
    std::lock_guard<std::mutex> lk(sensor_mu_);
    latest_targets_ = targets;
}

void Pipeline::updateIMU(const IMUData& imu) {
    std::lock_guard<std::mutex> lk(sensor_mu_);
    latest_imu_ = imu;
}

void Pipeline::updateGPS(const GPSData& gps, float heading) {
    std::lock_guard<std::mutex> lk(sensor_mu_);
    latest_gps_ = gps;
    latest_heading_ = heading;
}

} // namespace avm
