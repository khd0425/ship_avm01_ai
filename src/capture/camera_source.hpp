#pragma once

#include "src/config/app_config.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace avm::capture {

// A camera as seen by the CaptureManager. Implementations:
//   * SimulatedSource  - synthetic frames, no hardware (development / CI)
//   * OpenCvSource     - V4L2 (/dev/videoN) or GStreamer pipeline via cv::VideoCapture
//                        (built when AVM_WITH_OPENCV is defined; this is the
//                        production path on Jetson, e.g. nvarguscamerasrc / v4l2src)
// Vendor specific GMSL2 drivers (libargus / NvBuffer) can be added behind the
// same interface once the camera hardware is known (spec ch. 7, items 1-4).
class ICameraSource {
public:
    virtual ~ICameraSource() = default;

    virtual bool open() = 0;

    /// Blocking grab of the next frame into `dst` (rows `pitch` bytes apart,
    /// in the format/size given by the camera configuration). Returns false on
    /// a failure such as a pulled cable; the caller closes and re-opens.
    /// `timestamp_us` is on the util::steady_now_us() clock.
    virtual bool grab(std::uint8_t* dst, std::size_t pitch, std::uint64_t& timestamp_us) = 0;

    virtual void close() = 0;
    virtual const char* name() const = 0;
};

/// Create the source for a camera configuration (empty `source` => simulated).
std::unique_ptr<ICameraSource> make_camera_source(const config::CameraCfg& cfg, bool is_eo, int index);

} // namespace avm::capture
