#pragma once

// Per-frame view rendering: raw fisheye -> virtual-camera view (FR-4, FR-5).
//
// The pixel mapping is geo::ViewMapper<float> (geometry/view_math.hpp), the
// same code the host unit tests verify against double precision. It is
// evaluated per output pixel on the GPU, so preset transitions and
// IMU stabilisation need no LUT rebuild.

#include "src/geometry/view_math.hpp"

#include <cstdint>

namespace avm::render {

/// src: RGB8 fisheye (device-accessible pointer, e.g. mapped pinned memory).
/// dst: RGBA8 output. Pixels not visible in the fisheye are painted black.
/// `cuda_stream` is a cudaStream_t passed as void*. Returns a cudaError_t (0 = ok).
int launch_render_view_rgb8_to_rgba(const std::uint8_t* src, int src_pitch, int src_w, int src_h,
                                    std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                                    const geo::ViewMapper<float>& mapper, void* cuda_stream);

} // namespace avm::render
