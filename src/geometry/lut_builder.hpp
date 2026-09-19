#pragma once

// Host-side LUT generation + on-disk format (FR-2.2 "LUT file").
//
// A LUT maps every output pixel of a view to a source pixel of the fisheye
// image. Invalid (not visible) pixels hold (-1, -1); the CUDA remap kernels
// paint them black instead of clamping.

#include "src/geometry/view_math.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace avm::geo {

struct LutMap {
    int width{0}, height{0};          // output size
    int src_width{0}, src_height{0};  // fisheye source size
    std::vector<float> map_x, map_y;  // width*height each, row-major

    std::size_t pixel_count() const { return static_cast<std::size_t>(width) * height; }
    /// Fraction of output pixels that have a valid source (0..1).
    double valid_fraction() const;
};

/// Build the LUT for a view. Source coordinates outside [0,src-1] are invalid.
LutMap build_view_lut(const FisheyeModel& model, const CameraPose& pose, const ViewParams& view,
                      int out_w, int out_h, int src_w, int src_h);

/// Save/load a LUT as a single binary file:
///   "AVMLUT01" | u32 version | u32 out_w,out_h,src_w,src_h | u32 meta_len |
///   meta (UTF-8, free-form, e.g. calibration id) | float map_x[] | float map_y[] | u32 crc32
/// (little-endian). load_lut verifies magic, sizes and CRC.
bool save_lut(const std::string& path, const LutMap& lut, const std::string& metadata,
              std::string* err = nullptr);
bool load_lut(const std::string& path, LutMap& lut, std::string* metadata = nullptr,
              std::string* err = nullptr);

std::uint32_t crc32(const void* data, std::size_t size, std::uint32_t seed = 0);

} // namespace avm::geo
