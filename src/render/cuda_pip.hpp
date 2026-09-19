#pragma once

// IR picture-in-picture compositing (FR-3.1) with AGC (FR-3.2) on the GPU.

#include <cstdint>

namespace avm::render {

struct PipRect { int x, y, w, h; };

/// Composite a MONO8 IR frame into the RGBA output rectangle. `lut` is the
/// 256-entry AGC table in device memory. A thin border marks the window.
int launch_pip_mono8(const std::uint8_t* ir, int ir_pitch, int ir_w, int ir_h,
                     const std::uint8_t* lut,
                     std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                     const PipRect& rect, void* cuda_stream);

/// MONO16 variant: `lut` has 4096 entries indexed by (value >> 4).
int launch_pip_mono16(const std::uint16_t* ir, int ir_pitch_bytes, int ir_w, int ir_h,
                      const std::uint8_t* lut,
                      std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                      const PipRect& rect, void* cuda_stream);

/// RGB8 colour camera channel (normal-FOV visible camera used in an IR slot): no AGC.
int launch_pip_rgb8(const std::uint8_t* src, int src_pitch, int src_w, int src_h,
                    std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                    const PipRect& rect, void* cuda_stream);

/// "Channel inactive" placeholder (FR-1.4): dark window with a cross.
int launch_pip_inactive(std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                        const PipRect& rect, void* cuda_stream);

} // namespace avm::render
