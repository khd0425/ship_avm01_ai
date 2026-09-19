#include "alignment_engine.hpp"
#include "src/core/error_checks.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace avm {

// ─── Kernel: Homography Warp + Alpha Blend (RGB8) ──────────────────────────
//
// For each output pixel:
//   1. Look up whether it falls within the IR camera's projected FOV
//   2. Apply homography to find source IR coordinate
//   3. Bilinear sample the IR image
//   4. Alpha blend with the EO base image using the precomputed weight
//
__global__ void kernel_warp_blend_rgb8(
    const uint8_t* __restrict__ eo_base,  int eo_pitch,
    const uint8_t* __restrict__ ir_src,   int ir_pitch,
    uint8_t*       __restrict__ dst,      int dst_pitch,
    const float*   __restrict__ H,        // 3×3 homography
    const float*   __restrict__ weight_map,
    int out_w, int out_h,
    int ir_w, int ir_h,
    float alpha_global)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= out_w || row >= out_h) return;

    // Read per-pixel blend weight
    float weight = weight_map[row * out_w + col];
    if (weight < 1e-6f) {
        // No IR coverage: pass through EO pixel
        const uint8_t* src_row = eo_base + row * eo_pitch;
        uint8_t* dst_row = dst + row * dst_pitch;
        for (int c = 0; c < 3; ++c)
            dst_row[col * 3 + c] = src_row[col * 3 + c];
        return;
    }

    // Apply homography:  src = H * dst (in homogeneous coordinates)
    float w_inv = 1.0f / (H[6] * col + H[7] * row + H[8]);
    float sx = (H[0] * col + H[1] * row + H[2]) * w_inv;
    float sy = (H[3] * col + H[4] * row + H[5]) * w_inv;

    // Clamp to IR image bounds
    sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(ir_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(ir_h - 1));

    // Bilinear interpolation on IR
    int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    int x1 = min(x0 + 1, ir_w - 1), y1 = min(y0 + 1, ir_h - 1);
    float fx = sx - x0, fy = sy - y0;
    float fx1 = 1.0f - fx, fy1 = 1.0f - fy;

    const uint8_t* ir_row0 = ir_src + y0 * ir_pitch;
    const uint8_t* ir_row1 = ir_src + y1 * ir_pitch;
    const uint8_t* eo_row  = eo_base + row * eo_pitch;
    uint8_t* dst_row = dst + row * dst_pitch;

    // Blend each channel:  dst = (1 - a) * eo + a * ir
    // where a = weight * alpha_global, and we convert mono IR to RGB
    float a = weight * alpha_global;
    float one_minus_a = 1.0f - a;

    for (int c = 0; c < 3; ++c) {
        float ir_val = fx1 * fy1 * ir_row0[x0 * 3 + c]
                     + fx  * fy1 * ir_row0[x1 * 3 + c]
                     + fx1 * fy   * ir_row1[x0 * 3 + c]
                     + fx  * fy   * ir_row1[x1 * 3 + c];
        dst_row[col * 3 + c] = static_cast<uint8_t>(
            one_minus_a * eo_row[col * 3 + c] + a * ir_val);
    }
}

// ─── Kernel: Mono8 IR → RGB8 warp + blend ──────────────────────────────────

__global__ void kernel_warp_blend_ir_mono(
    const uint8_t* __restrict__ eo_base,  int eo_pitch,
    const uint8_t* __restrict__ ir_src,   int ir_pitch,
    uint8_t*       __restrict__ dst,      int dst_pitch,
    const float*   __restrict__ H,
    const float*   __restrict__ weight_map,
    int out_w, int out_h,
    int ir_w, int ir_h,
    float alpha_global)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= out_w || row >= out_h) return;

    float weight = weight_map[row * out_w + col];
    if (weight < 1e-6f) {
        const uint8_t* src_row = eo_base + row * eo_pitch;
        uint8_t* dst_row = dst + row * dst_pitch;
        for (int c = 0; c < 3; ++c)
            dst_row[col * 3 + c] = src_row[col * 3 + c];
        return;
    }

    float w_inv = 1.0f / (H[6] * col + H[7] * row + H[8]);
    float sx = (H[0] * col + H[1] * row + H[2]) * w_inv;
    float sy = (H[3] * col + H[4] * row + H[5]) * w_inv;

    sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(ir_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(ir_h - 1));

    int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    int x1 = min(x0 + 1, ir_w - 1), y1 = min(y0 + 1, ir_h - 1);
    float fx = sx - x0, fy = sy - y0;
    float fx1 = 1.0f - fx, fy1 = 1.0f - fy;

    float ir_val = fx1 * fy1 * ir_src[y0 * ir_pitch + x0]
                 + fx  * fy1 * ir_src[y0 * ir_pitch + x1]
                 + fx1 * fy   * ir_src[y1 * ir_pitch + x0]
                 + fx  * fy   * ir_src[y1 * ir_pitch + x1];

    // Blend with EO (IR as grayscale mapped to RGB)
    float a = weight * alpha_global;
    float one_minus_a = 1.0f - a;

    const uint8_t* eo_row = eo_base + row * eo_pitch;
    uint8_t* dst_row = dst + row * dst_pitch;

    // Thermal color mapping: ir_val → pseudo-color (hot = brighter/warmer)
    // Simple approach: blend ir_val equally into RGB with slight red tint
    float r = one_minus_a * eo_row[col * 3 + 0] + a * fminf(ir_val * 1.2f, 255.0f);
    float g = one_minus_a * eo_row[col * 3 + 1] + a * ir_val;
    float b = one_minus_a * eo_row[col * 3 + 2] + a * fmaxf(ir_val * 0.8f, 0.0f);

    dst_row[col * 3 + 0] = static_cast<uint8_t>(r);
    dst_row[col * 3 + 1] = static_cast<uint8_t>(g);
    dst_row[col * 3 + 2] = static_cast<uint8_t>(b);
}

// ─── Implementation ─────────────────────────────────────────────────────────

AlignmentEngine::AlignmentEngine(
    const CameraCalibration& eo_calib,
    const std::vector<CameraCalibration>& ir_calibs,
    uint32_t output_width,
    uint32_t output_height)
    : eo_calib_(eo_calib)
    , ir_calibs_(ir_calibs)
    , out_w_(output_width)
    , out_h_(output_height) {

    ir_maps_.resize(ir_calibs.size());
}

AlignmentEngine::~AlignmentEngine() {
    for (auto& m : ir_maps_) {
        if (m.d_homography) cudaFree(m.d_homography);
        if (m.d_weight_map) cudaFree(m.d_weight_map);
    }
    if (eo_weights_.d_weights) cudaFree(eo_weights_.d_weights);
}

bool AlignmentEngine::buildMaps(cudaStream_t stream) {
    if (maps_built_) return true;

    size_t num_pixels = static_cast<size_t>(out_w_) * out_h_;

    for (size_t i = 0; i < ir_calibs_.size(); ++i) {
        IRMap& m = ir_maps_[i];
        const auto& ir_calib = ir_calibs_[i];

        // ─── Compute homography H = K_eo * [R | t] * K_ir⁻¹ ────────────
        // This maps a pixel from the IR camera's undistorted view into
        // the EO camera's undistorted coordinate space.
        //
        // For simplicity, we construct a 3×3 homography from extrinsics.
        // In practice this would be pre-calibrated offline.
        //
        std::array<float, 9> H_host{};
        {
            // Simplified: identity homography + translation offset
            // (Real system would compose:
            //  H = K_eo * (R_eo_ir + t_eo_ir * n^T / d) * inv(K_ir))
            float tx = 0.0f, ty = 0.0f;
            if (i == 1) { tx = 0.05f; ty = -0.02f; } // IR camera offset example

            // 3×3 homography
            H_host[0] = 1.0f; H_host[1] = 0.0f; H_host[2] = tx * out_w_;
            H_host[3] = 0.0f; H_host[4] = 1.0f; H_host[5] = ty * out_h_;
            H_host[6] = 0.0f; H_host[7] = 0.0f; H_host[8] = 1.0f;
        }

        CUDA_CHECK(cudaMalloc(&m.d_homography, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(m.d_homography, H_host.data(),
                                    9 * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

        // ─── Build weight map (Gaussian falloff from center of projected ROI)
        std::vector<float> h_weights(num_pixels, 0.0f);
        for (uint32_t v = 0; v < out_h_; ++v) {
            for (uint32_t u = 0; u < out_w_; ++u) {
                // Apply homography to see if pixel maps to valid IR region
                float w_inv = 1.0f / (H_host[6] * u + H_host[7] * v + H_host[8]);
                float sx = (H_host[0] * u + H_host[1] * v + H_host[2]) * w_inv;
                float sy = (H_host[3] * u + H_host[4] * v + H_host[5]) * w_inv;

                if (sx >= 0 && sx < ir_calib.intr.fx * 2.0f &&  // approximate FOV
                    sy >= 0 && sy < ir_calib.intr.fy * 2.0f) {
                    // Weight = 1.0 in center, fall off towards edges
                    float nx = (sx / (ir_calib.intr.fx * 2.0f)) - 0.5f;
                    float ny = (sy / (ir_calib.intr.fy * 2.0f)) - 0.5f;
                    float dist2 = nx * nx + ny * ny;
                    h_weights[v * out_w_ + u] = expf(-dist2 * 4.0f);
                }
            }
        }

        CUDA_CHECK(cudaMalloc(&m.d_weight_map, num_pixels * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(m.d_weight_map, h_weights.data(),
                                    num_pixels * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        m.src_w = ir_calib.intr.fx * 2;  // approximate
        m.src_h = ir_calib.intr.fy * 2;
        m.valid = true;
    }

    // ─── EO weight map (constant 1.0) ───────────────────────────────────
    std::vector<float> h_eo_weights(num_pixels, 1.0f);
    CUDA_CHECK(cudaMalloc(&eo_weights_.d_weights, num_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(eo_weights_.d_weights, h_eo_weights.data(),
                                num_pixels * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    maps_built_ = true;
    return true;
}

void AlignmentEngine::blend(const CameraFrame& eo_frame,
                            const CameraFrame& ir_frame,
                            int ir_index,
                            CameraFrame& output,
                            cudaStream_t stream) {
    if (!maps_built_ || ir_index < 0 || ir_index >= static_cast<int>(ir_maps_.size()))
        return;

    const auto& m = ir_maps_[ir_index];
    if (!m.valid) return;

    dim3 block(32, 8);
    dim3 grid((out_w_ + block.x - 1) / block.x,
              (out_h_ + block.y - 1) / block.y);

    // Pick kernel based on IR pixel format
    if (ir_frame.spec.pixel_format == PixelFormat::MONO8 ||
        ir_frame.spec.pixel_format == PixelFormat::MONO16) {
        kernel_warp_blend_ir_mono<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(eo_frame.device_data),
            static_cast<int>(eo_frame.device_pitch),
            static_cast<const uint8_t*>(ir_frame.device_data),
            static_cast<int>(ir_frame.device_pitch),
            static_cast<uint8_t*>(output.device_data),
            static_cast<int>(output.device_pitch),
            m.d_homography, m.d_weight_map,
            static_cast<int>(out_w_), static_cast<int>(out_h_),
            static_cast<int>(ir_frame.spec.width),
            static_cast<int>(ir_frame.spec.height),
            0.35f);  // alpha global
    } else {
        kernel_warp_blend_rgb8<<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(eo_frame.device_data),
            static_cast<int>(eo_frame.device_pitch),
            static_cast<const uint8_t*>(ir_frame.device_data),
            static_cast<int>(ir_frame.device_pitch),
            static_cast<uint8_t*>(output.device_data),
            static_cast<int>(output.device_pitch),
            m.d_homography, m.d_weight_map,
            static_cast<int>(out_w_), static_cast<int>(out_h_),
            static_cast<int>(ir_frame.spec.width),
            static_cast<int>(ir_frame.spec.height),
            0.35f);
    }

    CUDA_CHECK(cudaGetLastError());
}

void AlignmentEngine::blendAll(const CameraFrame& eo_frame,
                               const std::vector<CameraFrame>& ir_frames,
                               CameraFrame& output,
                               cudaStream_t stream) {
    // Start with EO as base
    CUDA_CHECK(cudaMemcpy2DAsync(output.device_data, output.device_pitch,
                                  eo_frame.device_data, eo_frame.device_pitch,
                                  eo_frame.spec.width * 3, eo_frame.spec.height,
                                  cudaMemcpyDeviceToDevice, stream));

    // Blend each IR frame on top, chaining through a temp buffer if needed
    // For simplicity, blend directly into output
    for (int i = 0; i < static_cast<int>(ir_frames.size()) && i < static_cast<int>(ir_maps_.size()); ++i) {
        if (ir_maps_[i].valid && ir_frames[i].device_data) {
            blend(output, ir_frames[i], i, output, stream);
        }
    }
}

} // namespace avm
