#pragma once

#include "src/capture/camera_source.hpp"
#include "src/core/error_checks.hpp"
#include "src/core/types.hpp"
#include "src/sync/frame_sync.hpp"
#include "src/util/frame_ring.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace avm {

// ─── CaptureManager ─────────────────────────────────────────────────────────
//
// One acquisition thread per camera (1 EO + up to 3 IR). Frames are written
// into mapped pinned memory (cudaHostAlloc(cudaHostAllocMapped)): on Jetson the
// GPU reads the very same DRAM, so there is no host->device copy (zero-copy,
// spec ch. 1.2). Each channel has a 3-slot ring so the capture thread never
// blocks and a frame being processed is never overwritten (FrameRing).
//
// A camera that stops delivering (cable pulled) is re-opened once per second;
// the rest of the system keeps running (FR-1.4). The sync layer detects the
// stale channel from the missing timestamps.
//
class CaptureManager {
public:
    /// (channel, stamp, now_us). channel: -1 = EO, 0.. = IR index.
    using StampCallback = std::function<void(int, const sync::FrameStamp&, std::uint64_t)>;

    explicit CaptureManager(const PipelineConfig& config);
    ~CaptureManager();

    CaptureManager(const CaptureManager&) = delete;
    CaptureManager& operator=(const CaptureManager&) = delete;

    /// Allocate the mapped frame rings and create the camera sources.
    bool initialize();

    /// Start one acquisition thread per camera.
    bool start(StampCallback on_stamp);
    void stop();

    /// Wait for an EO frame newer than `after_seq`; nullptr on timeout.
    /// The frame stays valid until releaseEo().
    const CameraFrame* acquireEo(std::uint64_t after_seq, std::uint64_t& seq,
                                 std::chrono::milliseconds timeout);
    void releaseEo();

    /// Newest IR frame of channel `idx` (nullptr if none yet). Release when done.
    const CameraFrame* acquireIr(int idx, std::uint64_t& seq);

    /// IR frame closest in time to `target_us` among the recent history (nullptr if none).
    /// This is what the EO/IR time sync uses: the EO frame being processed is usually a
    /// few frames older than the newest IR frame. Release with releaseIr().
    const CameraFrame* acquireIrNearest(int idx, std::uint64_t target_us, std::uint64_t& seq);
    void releaseIr(int idx);

    int num_ir_cameras() const { return static_cast<int>(channels_.size()) - 1; }
    bool link_up(int channel) const;   // channel: -1 = EO, 0.. = IR

private:
    struct Channel {
        CameraSpec spec{};
        std::unique_ptr<capture::ICameraSource> source;
        std::vector<MappedBuffer> buffers;
        std::vector<CameraFrame> frames;
        std::unique_ptr<util::FrameRing> ring;
        std::size_t held_slot{0};
        std::thread thread;
        std::atomic<bool> link_up{false};
    };

    bool allocate(Channel& ch, const CameraSpec& spec, std::size_t slots);
    void captureLoop(Channel& ch, int channel_id);
    Channel& channel(int id) { return *channels_.at(static_cast<std::size_t>(id + 1)); }
    const Channel& channel(int id) const { return *channels_.at(static_cast<std::size_t>(id + 1)); }

    PipelineConfig config_;
    std::vector<std::unique_ptr<Channel>> channels_;   // [0] = EO, [1..] = IR
    StampCallback on_stamp_;
    std::atomic<bool> running_{false};
    bool initialized_{false};
};

} // namespace avm
