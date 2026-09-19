#pragma once

// Slot bookkeeping for one capture thread (writer) and one processing thread
// (reader) sharing N pre-allocated frame buffers (no allocation at runtime).
//
//   writer: slot = begin_write();  ...fill buffer...;  end_write(slot, seq);
//   reader: wait_new(slot, seq, last_seq, timeout);  ...process...;  release(slot);
//
// With 3 slots there is always one slot that is neither the latest published
// frame nor held by the reader, so the writer never blocks and never
// overwrites a frame that is being processed. A slow reader simply skips
// frames and always gets the newest one (lowest latency, PR-3).
//
// The writer always recycles the OLDEST published slot, so with N slots the
// last N-2 frames stay available; acquire_nearest() lets the reader pick the
// frame closest in time to a reference (used to match IR frames to an EO frame
// that is still being processed a few frames after it arrived).

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace avm::util {

class FrameRing {
public:
    explicit FrameRing(std::size_t slots = 3) : slots_(slots < 3 ? 3 : slots), seq_(slots_, 0) {}

    std::size_t slots() const { return slots_; }

    /// Slot to fill next: the oldest one that is not held by the reader.
    std::size_t begin_write() {
        std::lock_guard<std::mutex> lk(mu_);
        std::size_t best = slots_;
        for (std::size_t s = 0; s < slots_; ++s) {
            if (held_valid_ && s == held_) continue;
            if (have_latest_ && s == latest_) continue;
            if (best == slots_ || seq_[s] < seq_[best]) best = s;
        }
        if (best == slots_) best = 0;   // unreachable with >= 3 slots
        writing_ = best;
        writing_valid_ = true;
        seq_[best] = 0;                 // contents are being overwritten: not selectable
        return best;
    }

    void end_write(std::size_t slot, std::uint64_t seq) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            latest_ = slot;
            latest_seq_ = seq;
            seq_[slot] = seq;
            have_latest_ = true;
            writing_valid_ = false;
        }
        cv_.notify_all();
    }

    /// Hold the published frame with the smallest cost(slot) (e.g. |timestamp - target|).
    /// `cost` runs under the ring's lock and must be cheap. False if nothing is published.
    bool acquire_nearest(std::size_t& slot, std::uint64_t& seq,
                         const std::function<std::int64_t(std::size_t)>& cost) {
        std::lock_guard<std::mutex> lk(mu_);
        bool found = false;
        std::int64_t best_cost = 0;
        std::size_t best = 0;
        for (std::size_t s = 0; s < slots_; ++s) {
            if (seq_[s] == 0) continue;
            if (writing_valid_ && s == writing_) continue;
            const std::int64_t c = cost(s);
            if (!found || c < best_cost) { found = true; best_cost = c; best = s; }
        }
        if (!found) return false;
        slot = held_ = best;
        seq = seq_[best];
        held_valid_ = true;
        return true;
    }

    /// Newest published frame (marks it held). False if nothing published yet.
    bool acquire_latest(std::size_t& slot, std::uint64_t& seq) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!have_latest_) return false;
        slot = held_ = latest_;
        seq = latest_seq_;
        held_valid_ = true;
        return true;
    }

    /// Wait until a frame with seq > after_seq exists, then hold and return it.
    bool wait_new(std::size_t& slot, std::uint64_t& seq, std::uint64_t after_seq,
                  std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_.wait_for(lk, timeout, [&] { return have_latest_ && latest_seq_ > after_seq; }))
            return false;
        slot = held_ = latest_;
        seq = latest_seq_;
        held_valid_ = true;
        return true;
    }

    void release(std::size_t slot) {
        std::lock_guard<std::mutex> lk(mu_);
        if (held_valid_ && held_ == slot) held_valid_ = false;
    }

    /// Wake a waiting reader (used on shutdown).
    void wake() { cv_.notify_all(); }

private:
    std::size_t slots_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::uint64_t> seq_;      // sequence number stored in each slot (0 = empty/being written)
    std::size_t latest_{0}, held_{0}, writing_{0};
    std::uint64_t latest_seq_{0};
    bool have_latest_{false}, held_valid_{false}, writing_valid_{false};
};

} // namespace avm::util
