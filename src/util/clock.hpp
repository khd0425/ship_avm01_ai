#pragma once

#include <chrono>
#include <cstdint>

namespace avm::util {

/// Monotonic clock in microseconds. All capture/sync timestamps use this clock.
inline std::uint64_t steady_now_us() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline double steady_now_ms() { return static_cast<double>(steady_now_us()) / 1000.0; }

} // namespace avm::util
