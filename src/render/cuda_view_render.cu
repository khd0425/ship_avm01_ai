#include "cuda_view_render.hpp"

#include <cuda_runtime.h>

namespace avm::render {

__global__ void kernel_render_view_rgb8_to_rgba(
    const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
    uint8_t* __restrict__ dst, int dst_pitch, int dst_w, int dst_h,
    geo::ViewMapper<float> mapper)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= dst_w || row >= dst_h) return;

    uint8_t* px = dst + row * dst_pitch + col * 4;

    float sx, sy;
    if (!mapper.map(static_cast<float>(col), static_cast<float>(row), sx, sy) ||
        sx < 0.0f || sy < 0.0f || sx > static_cast<float>(src_w - 1) ||
        sy > static_cast<float>(src_h - 1)) {
        px[0] = 0; px[1] = 0; px[2] = 0; px[3] = 255;
        return;
    }

    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const uint8_t* r0 = src + y0 * src_pitch;
    const uint8_t* r1 = src + y1 * src_pitch;
    for (int c = 0; c < 3; ++c) {
        const float v = w00 * r0[x0 * 3 + c] + w10 * r0[x1 * 3 + c]
                      + w01 * r1[x0 * 3 + c] + w11 * r1[x1 * 3 + c];
        px[c] = static_cast<uint8_t>(v + 0.5f);
    }
    px[3] = 255;
}

int launch_render_view_rgb8_to_rgba(const std::uint8_t* src, int src_pitch, int src_w, int src_h,
                                    std::uint8_t* dst, int dst_pitch, int dst_w, int dst_h,
                                    const geo::ViewMapper<float>& mapper, void* cuda_stream)
{
    const dim3 block(32, 8);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    kernel_render_view_rgb8_to_rgba<<<grid, block, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
        src, src_pitch, src_w, src_h, dst, dst_pitch, dst_w, dst_h, mapper);
    return static_cast<int>(cudaGetLastError());
}

} // namespace avm::render
