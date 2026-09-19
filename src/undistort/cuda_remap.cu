#include "undistort_engine.hpp"
#include "src/core/error_checks.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>

namespace avm {

// ─── Kernel: Fisheye Undistortion Remap (Bilinear) ─────────────────────────
//
// Each thread handles one output pixel.  We look up the source coordinate
// from the precomputed LUTs and perform bilinear interpolation.
//
// For fisheye → pinhole rectification, the LUT maps (x_out, y_out) → (x_in, y_in).
//
__global__ void kernel_remap_rgb8(
    const uint8_t* __restrict__ src,   int src_pitch,
    uint8_t*       __restrict__ dst,   int dst_pitch,
    const float*   __restrict__ map_x,
    const float*   __restrict__ map_y,
    int out_w, int out_h,
    int src_w, int src_h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= out_w || row >= out_h) return;

    // Look up source coordinate
    float sx = map_x[row * out_w + col];
    float sy = map_y[row * out_w + col];

    // Invalid LUT entries (outside the lens FOV / image) are marked negative by
    // the LUT builder: paint them black instead of smearing edge pixels.
    uint8_t* dst_px = dst + row * dst_pitch + col * 3;
    if (sx < 0.0f || sy < 0.0f || sx > static_cast<float>(src_w - 1) ||
        sy > static_cast<float>(src_h - 1)) {
        dst_px[0] = dst_px[1] = dst_px[2] = 0;
        return;
    }

    // Bilinear interpolation
    int x0 = static_cast<int>(sx);
    int y0 = static_cast<int>(sy);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = sx - x0;
    float fy = sy - y0;
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;

    const uint8_t* src_row0 = src + y0 * src_pitch;
    const uint8_t* src_row1 = src + y1 * src_pitch;
    uint8_t* dst_row = dst + row * dst_pitch;

    for (int c = 0; c < 3; ++c) {
        float v = fx1 * fy1 * src_row0[x0 * 3 + c]
                + fx  * fy1 * src_row0[x1 * 3 + c]
                + fx1 * fy   * src_row1[x0 * 3 + c]
                + fx  * fy   * src_row1[x1 * 3 + c];
        dst_row[col * 3 + c] = static_cast<uint8_t>(v);
    }
}

// ─── Kernel: Mono8 Remap (for IR thermal) ──────────────────────────────────

__global__ void kernel_remap_mono8(
    const uint8_t* __restrict__ src,   int src_pitch,
    uint8_t*       __restrict__ dst,   int dst_pitch,
    const float*   __restrict__ map_x,
    const float*   __restrict__ map_y,
    int out_w, int out_h,
    int src_w, int src_h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= out_w || row >= out_h) return;

    float sx = map_x[row * out_w + col];
    float sy = map_y[row * out_w + col];
    if (sx < 0.0f || sy < 0.0f || sx > static_cast<float>(src_w - 1) ||
        sy > static_cast<float>(src_h - 1)) {
        dst[row * dst_pitch + col] = 0;
        return;
    }

    int x0 = static_cast<int>(sx);
    int y0 = static_cast<int>(sy);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float fx = sx - x0, fy = sy - y0;
    float fx1 = 1.0f - fx, fy1 = 1.0f - fy;

    float v = fx1 * fy1 * src[y0 * src_pitch + x0]
            + fx  * fy1 * src[y0 * src_pitch + x1]
            + fx1 * fy   * src[y1 * src_pitch + x0]
            + fx  * fy   * src[y1 * src_pitch + x1];

    dst[row * dst_pitch + col] = static_cast<uint8_t>(v);
}

// ─── Implementation ─────────────────────────────────────────────────────────

UndistortEngine::UndistortEngine(const geo::FisheyeModel& model, const geo::CameraPose& pose,
                                 const geo::ViewParams& view,
                                 uint32_t output_width, uint32_t output_height,
                                 uint32_t src_width, uint32_t src_height)
    : model_(model)
    , pose_(pose)
    , view_(view)
    , out_width_(output_width)
    , out_height_(output_height)
    , src_width_(src_width)
    , src_height_(src_height) {
}

bool UndistortEngine::buildLUTs(cudaStream_t stream) {
    if (luts_built_) return true;
    // The calibration -> LUT math lives in geometry/ (host, unit-tested); the
    // same view mapping is used by the per-frame renderer.
    geo::LutMap lut = geo::build_view_lut(model_, pose_, view_,
                                          static_cast<int>(out_width_), static_cast<int>(out_height_),
                                          static_cast<int>(src_width_), static_cast<int>(src_height_));
    return uploadLUT(lut, stream);
}

bool UndistortEngine::uploadLUT(const geo::LutMap& lut, cudaStream_t stream) {
    if (static_cast<uint32_t>(lut.width) != out_width_ || static_cast<uint32_t>(lut.height) != out_height_ ||
        lut.map_x.size() != lut.pixel_count() || lut.map_y.size() != lut.pixel_count()) {
        return false;
    }
    const size_t bytes = lut.pixel_count() * sizeof(float);
    d_map_x_ = alloc_device<float>(lut.pixel_count());
    d_map_y_ = alloc_device<float>(lut.pixel_count());
    CUDA_CHECK(cudaMemcpyAsync(d_map_x_.get(), lut.map_x.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_map_y_.get(), lut.map_y.data(), bytes, cudaMemcpyHostToDevice, stream));
    // The host vectors go out of scope after return: wait for the copies.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    luts_built_ = true;
    return true;
}

void UndistortEngine::correct(const CameraFrame& input,
                              CameraFrame& output,
                              cudaStream_t stream) {
    if (!luts_built_) return;

    dim3 block(32, 8);
    dim3 grid((out_width_  + block.x - 1) / block.x,
              (out_height_ + block.y - 1) / block.y);

    const float* map_x = static_cast<const float*>(d_map_x_.get());
    const float* map_y = static_cast<const float*>(d_map_y_.get());

    if (input.spec.pixel_format == PixelFormat::RGB8 ||
        input.spec.pixel_format == PixelFormat::RGBA8) {
        kernel_remap_rgb8<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(input.device_data),
            static_cast<int>(input.device_pitch),
            static_cast<uint8_t*>(output.device_data),
            static_cast<int>(output.device_pitch),
            map_x, map_y,
            static_cast<int>(out_width_),
            static_cast<int>(out_height_),
            static_cast<int>(input.spec.width),
            static_cast<int>(input.spec.height));
    } else {
        // MONO8 or other single-channel
        kernel_remap_mono8<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(input.device_data),
            static_cast<int>(input.device_pitch),
            static_cast<uint8_t*>(output.device_data),
            static_cast<int>(output.device_pitch),
            map_x, map_y,
            static_cast<int>(out_width_),
            static_cast<int>(out_height_),
            static_cast<int>(input.spec.width),
            static_cast<int>(input.spec.height));
    }

    CUDA_CHECK(cudaGetLastError());  // catch kernel launch errors
}

} // namespace avm
