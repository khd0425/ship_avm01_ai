#include "cuda_pip.hpp"

#include <cuda_runtime.h>

namespace avm::render {

namespace {

constexpr int kBorder = 2;

__device__ inline bool in_border(int col, int row, int w, int h) {
    return col < kBorder || row < kBorder || col >= w - kBorder || row >= h - kBorder;
}

__global__ void kernel_pip_mono8(
    const uint8_t* __restrict__ ir, int ir_pitch, int ir_w, int ir_h,
    const uint8_t* __restrict__ lut,
    uint8_t* __restrict__ dst, int dst_pitch, int dst_w, int dst_h,
    PipRect rect)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= rect.w || row >= rect.h) return;
    const int x = rect.x + col, y = rect.y + row;
    if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) return;

    uint8_t* px = dst + y * dst_pitch + x * 4;
    if (in_border(col, row, rect.w, rect.h)) {
        px[0] = 0; px[1] = 200; px[2] = 255; px[3] = 255;
        return;
    }

    // bilinear sample of the raw IR value, then AGC table
    float sx = (col + 0.5f) * ir_w / rect.w - 0.5f;
    float sy = (row + 0.5f) * ir_h / rect.h - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(ir_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(ir_h - 1));
    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, ir_w - 1), y1 = min(y0 + 1, ir_h - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float v = (1.0f - fx) * (1.0f - fy) * ir[y0 * ir_pitch + x0]
                  + fx * (1.0f - fy) * ir[y0 * ir_pitch + x1]
                  + (1.0f - fx) * fy * ir[y1 * ir_pitch + x0]
                  + fx * fy * ir[y1 * ir_pitch + x1];
    const uint8_t g = lut[static_cast<int>(v + 0.5f)];
    px[0] = g; px[1] = g; px[2] = g; px[3] = 255;
}

__global__ void kernel_pip_mono16(
    const uint16_t* __restrict__ ir, int ir_pitch_bytes, int ir_w, int ir_h,
    const uint8_t* __restrict__ lut,
    uint8_t* __restrict__ dst, int dst_pitch, int dst_w, int dst_h,
    PipRect rect)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= rect.w || row >= rect.h) return;
    const int x = rect.x + col, y = rect.y + row;
    if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) return;

    uint8_t* px = dst + y * dst_pitch + x * 4;
    if (in_border(col, row, rect.w, rect.h)) {
        px[0] = 0; px[1] = 200; px[2] = 255; px[3] = 255;
        return;
    }
    // nearest sample (16-bit values are not interpolated before the table)
    const int sx = min(ir_w - 1, static_cast<int>((col + 0.5f) * ir_w / rect.w));
    const int sy = min(ir_h - 1, static_cast<int>((row + 0.5f) * ir_h / rect.h));
    const uint16_t* r = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const uint8_t*>(ir) + sy * ir_pitch_bytes);
    const uint8_t g = lut[r[sx] >> 4];
    px[0] = g; px[1] = g; px[2] = g; px[3] = 255;
}

__global__ void kernel_pip_rgb8(
    const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
    uint8_t* __restrict__ dst, int dst_pitch, int dst_w, int dst_h,
    PipRect rect)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= rect.w || row >= rect.h) return;
    const int x = rect.x + col, y = rect.y + row;
    if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) return;

    uint8_t* px = dst + y * dst_pitch + x * 4;
    if (in_border(col, row, rect.w, rect.h)) {
        px[0] = 0; px[1] = 200; px[2] = 255; px[3] = 255;
        return;
    }
    float sx = (col + 0.5f) * src_w / rect.w - 0.5f;
    float sy = (row + 0.5f) * src_h / rect.h - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(src_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(src_h - 1));
    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    const float fx = sx - x0, fy = sy - y0;
    const uint8_t* r0 = src + y0 * src_pitch;
    const uint8_t* r1 = src + y1 * src_pitch;
    for (int c = 0; c < 3; ++c) {
        const float v = (1.0f - fx) * (1.0f - fy) * r0[x0 * 3 + c] + fx * (1.0f - fy) * r0[x1 * 3 + c]
                      + (1.0f - fx) * fy * r1[x0 * 3 + c] + fx * fy * r1[x1 * 3 + c];
        px[c] = static_cast<uint8_t>(v + 0.5f);
    }
    px[3] = 255;
}

__global__ void kernel_pip_inactive(uint8_t* __restrict__ dst, int dst_pitch, int dst_w, int dst_h,
                                    PipRect rect)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= rect.w || row >= rect.h) return;
    const int x = rect.x + col, y = rect.y + row;
    if (x < 0 || y < 0 || x >= dst_w || y >= dst_h) return;

    uint8_t* px = dst + y * dst_pitch + x * 4;
    if (in_border(col, row, rect.w, rect.h)) {
        px[0] = 200; px[1] = 40; px[2] = 40; px[3] = 255;      // red frame = inactive
        return;
    }
    // diagonal cross
    const float u = static_cast<float>(col) / rect.w;
    const float v = static_cast<float>(row) / rect.h;
    const bool cross = fabsf(u - v) < 0.012f || fabsf(u + v - 1.0f) < 0.012f;
    const uint8_t g = cross ? 160 : 24;
    px[0] = g; px[1] = g; px[2] = g; px[3] = 255;
}

inline dim3 grid_for(const PipRect& r, const dim3& block) {
    return dim3((r.w + block.x - 1) / block.x, (r.h + block.y - 1) / block.y);
}

} // namespace

int launch_pip_mono8(const std::uint8_t* ir, int ir_pitch, int ir_w, int ir_h,
                     const std::uint8_t* lut,
                     std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                     const PipRect& rect, void* cuda_stream)
{
    if (rect.w <= 0 || rect.h <= 0) return 0;
    const dim3 block(16, 16);
    kernel_pip_mono8<<<grid_for(rect, block), block, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
        ir, ir_pitch, ir_w, ir_h, lut, dst, dst_pitch, dst_w, dst_h, rect);
    return static_cast<int>(cudaGetLastError());
}

int launch_pip_mono16(const std::uint16_t* ir, int ir_pitch_bytes, int ir_w, int ir_h,
                      const std::uint8_t* lut,
                      std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                      const PipRect& rect, void* cuda_stream)
{
    if (rect.w <= 0 || rect.h <= 0) return 0;
    const dim3 block(16, 16);
    kernel_pip_mono16<<<grid_for(rect, block), block, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
        ir, ir_pitch_bytes, ir_w, ir_h, lut, dst, dst_pitch, dst_w, dst_h, rect);
    return static_cast<int>(cudaGetLastError());
}

int launch_pip_rgb8(const std::uint8_t* src, int src_pitch, int src_w, int src_h,
                    std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                    const PipRect& rect, void* cuda_stream)
{
    if (rect.w <= 0 || rect.h <= 0) return 0;
    const dim3 block(16, 16);
    kernel_pip_rgb8<<<grid_for(rect, block), block, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
        src, src_pitch, src_w, src_h, dst, dst_pitch, dst_w, dst_h, rect);
    return static_cast<int>(cudaGetLastError());
}

int launch_pip_inactive(std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                        const PipRect& rect, void* cuda_stream)
{
    if (rect.w <= 0 || rect.h <= 0) return 0;
    const dim3 block(16, 16);
    kernel_pip_inactive<<<grid_for(rect, block), block, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
        dst, dst_pitch, dst_w, dst_h, rect);
    return static_cast<int>(cudaGetLastError());
}

} // namespace avm::render
