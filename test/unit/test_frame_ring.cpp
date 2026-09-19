#include "test_util.hpp"

#include "src/util/frame_ring.hpp"

#include <atomic>
#include <thread>
#include <vector>

using avm::util::FrameRing;
using namespace std::chrono_literals;

static void test_basic() {
    FrameRing r(3);
    std::size_t slot; std::uint64_t seq;
    CHECK(!r.acquire_latest(slot, seq));                 // nothing yet
    CHECK(!r.wait_new(slot, seq, 0, 5ms));

    std::size_t w = r.begin_write();
    r.end_write(w, 1);
    CHECK(r.acquire_latest(slot, seq) && slot == w && seq == 1);

    // While the reader holds `w` and `w` is latest, the writer gets another slot.
    std::size_t w2 = r.begin_write();
    CHECK(w2 != w);
    r.end_write(w2, 2);
    // Reader still holds w (old frame): writer must avoid both w and w2.
    std::size_t w3 = r.begin_write();
    CHECK(w3 != w && w3 != w2);
    r.release(w);

    CHECK(r.wait_new(slot, seq, 1, 5ms) && slot == w2 && seq == 2);
    CHECK(!r.wait_new(slot, seq, 2, 5ms));               // no newer frame -> timeout
}

static void test_history_and_nearest() {
    FrameRing r(6);
    std::vector<std::int64_t> ts(6, 0);
    std::size_t slot; std::uint64_t seq;
    CHECK(!r.acquire_nearest(slot, seq, [&](std::size_t) { return std::int64_t(0); }));   // empty

    for (int i = 1; i <= 9; ++i) {                        // 9 frames into 6 slots, 33 ms apart
        std::size_t w = r.begin_write();
        ts[w] = i * 33;
        r.end_write(w, static_cast<std::uint64_t>(i));
    }
    // frames 4..9 are still there (frames 1..3 were recycled, oldest first)
    auto nearest = [&](std::int64_t target) {
        CHECK(r.acquire_nearest(slot, seq, [&](std::size_t s) { return std::llabs(ts[s] - target); }));
        std::uint64_t got = seq;
        r.release(slot);
        return got;
    };
    CHECK(nearest(9 * 33) == 9);
    CHECK(nearest(7 * 33 + 5) == 7);                     // an older frame is reachable
    CHECK(nearest(4 * 33) == 4);
    CHECK(nearest(1 * 33) == 4);                         // frame 1 is gone: closest remaining is 4

    // A held frame is never recycled, the slot being written is never selectable.
    CHECK(r.acquire_nearest(slot, seq, [&](std::size_t s) { return std::llabs(ts[s] - 5 * 33); }));
    CHECK(seq == 5);
    const std::size_t held = slot;
    std::size_t w2 = r.begin_write();
    CHECK(w2 != held);
    CHECK(!r.acquire_nearest(slot, seq, [&](std::size_t s) { return s == w2 ? std::int64_t(-1000000) : std::int64_t(0); })
          || slot != w2);                                // the slot under write is skipped
    r.end_write(w2, 10);
    r.release(held);
}

static void test_threads_never_tear() {
    // Writer fills a slot with a repeated value; the reader checks that every
    // frame it holds stays internally consistent while the writer keeps going.
    constexpr int kFrames = 4000, kSize = 256;
    FrameRing ring(3);
    std::vector<std::vector<int>> buf(3, std::vector<int>(kSize, 0));
    std::atomic<bool> torn{false}, done{false};

    std::thread writer([&] {
        for (int i = 1; i <= kFrames; ++i) {
            std::size_t s = ring.begin_write();
            for (int k = 0; k < kSize; ++k) buf[s][k] = i;
            ring.end_write(s, static_cast<std::uint64_t>(i));
            if (i % 50 == 0) std::this_thread::yield();
        }
        done = true;
    });

    std::uint64_t last = 0;
    int frames_seen = 0;
    while (!(done && last == kFrames)) {
        std::size_t s; std::uint64_t seq;
        if (!ring.wait_new(s, seq, last, 20ms)) { if (done) break; continue; }
        const int v = buf[s][0];
        for (int rep = 0; rep < 3; ++rep) {              // slow reader: several passes
            for (int k = 0; k < kSize; ++k) if (buf[s][k] != v) torn = true;
        }
        if (v != static_cast<int>(seq)) torn = true;
        CHECK(seq > last);                               // strictly newer, never goes back
        last = seq;
        ++frames_seen;
        ring.release(s);
    }
    writer.join();
    CHECK(!torn);
    CHECK(frames_seen > 0);
}

int main() {
    test_basic();
    test_history_and_nearest();
    test_threads_never_tear();
    TEST_MAIN_END("test_frame_ring");
}
