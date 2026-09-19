#include "capture_manager.hpp"

#include "src/util/clock.hpp"
#include "src/util/logger.hpp"

namespace avm {

CaptureManager::CaptureManager(const PipelineConfig& config) : config_(config) {}

CaptureManager::~CaptureManager() { stop(); }

static std::size_t bytes_per_pixel(PixelFormat f) {
    switch (f) {
        case PixelFormat::RGB8:   return 3;
        case PixelFormat::RGBA8:  return 4;
        case PixelFormat::MONO8:  return 1;
        case PixelFormat::MONO16: return 2;
        case PixelFormat::YUV420: return 1;   // Y plane only
    }
    return 1;
}

bool CaptureManager::allocate(Channel& ch, const CameraSpec& spec, std::size_t slots) {
    ch.spec = spec;
    ch.ring = std::make_unique<util::FrameRing>(slots);
    // 256-byte row alignment (NVIDIA surface requirement)
    const std::size_t pitch = ((static_cast<std::size_t>(spec.width) * bytes_per_pixel(spec.pixel_format) + 255) / 256) * 256;
    const std::size_t total = pitch * spec.height;
    ch.buffers.clear();
    ch.frames.assign(ch.ring->slots(), CameraFrame{});
    for (std::size_t s = 0; s < ch.ring->slots(); ++s) {
        ch.buffers.push_back(alloc_mapped(total));
        CameraFrame& f = ch.frames[s];
        f.spec = spec;
        f.host_data = ch.buffers[s].host.get();
        f.host_pitch = pitch;
        f.device_data = ch.buffers[s].device;   // same memory as host_data: zero-copy
        f.device_pitch = pitch;
    }
    return true;
}

bool CaptureManager::initialize() {
    if (initialized_) return true;

    try {
        auto eo = std::make_unique<Channel>();
        allocate(*eo, config_.eo_spec, 3);   // EO: 3 slots (latest, held, writing)
        eo->source = capture::make_camera_source(config_.app.eo, true, 0);
        channels_.push_back(std::move(eo));

        for (std::size_t i = 0; i < config_.ir_specs.size(); ++i) {
            auto ir = std::make_unique<Channel>();
            allocate(*ir, config_.ir_specs[i], 8);   // IR: 8 slots = ~6 frames of history for EO/IR matching
            ir->source = capture::make_camera_source(config_.app.ir.at(i), false, static_cast<int>(i));
            channels_.push_back(std::move(ir));
        }
    } catch (const std::exception& e) {
        AVM_LOGE("capture", "buffer allocation failed: %s", e.what());
        channels_.clear();
        return false;
    }
    initialized_ = true;
    return true;
}

bool CaptureManager::start(StampCallback on_stamp) {
    if (!initialized_) {
        AVM_LOGE("capture", "start() before initialize()");
        return false;
    }
    if (running_) return true;
    on_stamp_ = std::move(on_stamp);
    running_ = true;
    for (std::size_t i = 0; i < channels_.size(); ++i) {
        const int id = static_cast<int>(i) - 1;
        channels_[i]->thread = std::thread([this, id] { captureLoop(channel(id), id); });
    }
    AVM_LOGI("capture", "streaming started (1 EO + %d IR)", num_ir_cameras());
    return true;
}

void CaptureManager::stop() {
    if (!running_.exchange(false)) return;
    for (auto& ch : channels_) ch->ring->wake();
    for (auto& ch : channels_) if (ch->thread.joinable()) ch->thread.join();
    AVM_LOGI("capture", "streaming stopped");
}

void CaptureManager::captureLoop(Channel& ch, int channel_id) {
    const char* label = channel_id < 0 ? "EO" : "IR";
    std::uint64_t seq = 0;
    bool opened = false;
    bool warned = false;

    while (running_) {
        if (!opened) {
            if (ch.source->open()) {
                opened = true;
                warned = false;
                ch.link_up = true;
                AVM_LOGI("capture", "%s%d opened (%s) %ux%u @ %.1f fps", label,
                         channel_id < 0 ? 0 : channel_id, ch.source->name(),
                         ch.spec.width, ch.spec.height, ch.spec.fps);
            } else {
                ch.link_up = false;
                if (!warned) {
                    AVM_LOGW("capture", "%s%d not available; retrying every second", label,
                             channel_id < 0 ? 0 : channel_id);
                    warned = true;
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
        }

        const std::size_t slot = ch.ring->begin_write();
        CameraFrame& f = ch.frames[slot];
        std::uint64_t ts = 0;
        if (!ch.source->grab(static_cast<std::uint8_t*>(f.host_data), f.host_pitch, ts)) {
            AVM_LOGW("capture", "%s%d: grab failed - link down, re-opening", label,
                     channel_id < 0 ? 0 : channel_id);
            ch.source->close();
            opened = false;
            ch.link_up = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        f.timestamp_us = ts;
        f.frame_id = static_cast<std::uint32_t>(++seq);
        ch.ring->end_write(slot, seq);
        if (on_stamp_) on_stamp_(channel_id, sync::FrameStamp{ts, f.frame_id}, util::steady_now_us());
    }
    ch.source->close();
}

const CameraFrame* CaptureManager::acquireEo(std::uint64_t after_seq, std::uint64_t& seq,
                                             std::chrono::milliseconds timeout) {
    Channel& ch = channel(-1);
    std::size_t slot;
    if (!ch.ring->wait_new(slot, seq, after_seq, timeout)) return nullptr;
    ch.held_slot = slot;
    return &ch.frames[slot];
}

void CaptureManager::releaseEo() {
    Channel& ch = channel(-1);
    ch.ring->release(ch.held_slot);
}

const CameraFrame* CaptureManager::acquireIr(int idx, std::uint64_t& seq) {
    if (idx < 0 || idx >= num_ir_cameras()) return nullptr;
    Channel& ch = channel(idx);
    std::size_t slot;
    if (!ch.ring->acquire_latest(slot, seq)) return nullptr;
    ch.held_slot = slot;
    return &ch.frames[slot];
}

const CameraFrame* CaptureManager::acquireIrNearest(int idx, std::uint64_t target_us, std::uint64_t& seq) {
    if (idx < 0 || idx >= num_ir_cameras()) return nullptr;
    Channel& ch = channel(idx);
    std::size_t slot;
    const bool ok = ch.ring->acquire_nearest(slot, seq, [&](std::size_t s) {
        const std::int64_t d = static_cast<std::int64_t>(ch.frames[s].timestamp_us) - static_cast<std::int64_t>(target_us);
        return d < 0 ? -d : d;
    });
    if (!ok) return nullptr;
    ch.held_slot = slot;
    return &ch.frames[slot];
}

void CaptureManager::releaseIr(int idx) {
    if (idx < 0 || idx >= num_ir_cameras()) return;
    Channel& ch = channel(idx);
    ch.ring->release(ch.held_slot);
}

bool CaptureManager::link_up(int channel_id) const {
    if (channel_id < -1 || channel_id >= num_ir_cameras()) return false;
    return channel(channel_id).link_up.load();
}

} // namespace avm
