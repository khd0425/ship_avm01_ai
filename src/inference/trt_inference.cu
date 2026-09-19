#include "inference_engine.hpp"
#include "src/core/error_checks.hpp"
#include "include/avm/avm_config.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace avm {

// ─── Kernel: RGB8 → Float32 Normalized Preprocess ──────────────────────────
//
// Converts uint8 RGB to normalized float32 input for the TensorRT model.
// Layout: NCHW (batch=1, channels=3, height, width).
// Normalization:  pixel / 255.0f  (or mean/std if required).
//
__global__ void kernel_preprocess_rgb8_to_float(
    const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
    float*         __restrict__ dst, int dst_w, int dst_h,
    float scale_x, float scale_y)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= dst_w || row >= dst_h) return;

    // Map destination pixel to source (with scaling)
    float sx = (col + 0.5f) * scale_x - 0.5f;
    float sy = (row + 0.5f) * scale_y - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(src_w - 1));
    sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(src_h - 1));

    int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;
    float fx1 = 1.0f - fx, fy1 = 1.0f - fy;

    const uint8_t* row0 = src + y0 * src_pitch;
    const uint8_t* row1 = src + y1 * src_pitch;

    // NCHW layout: dst[0*H*W + row*W + col] = R
    //              dst[1*H*W + row*W + col] = G
    //              dst[2*H*W + row*W + col] = B
    int idx = row * dst_w + col;
    for (int c = 0; c < 3; ++c) {
        float v = fx1 * fy1 * row0[x0 * 3 + c]
                + fx  * fy1 * row0[x1 * 3 + c]
                + fx1 * fy   * row1[x0 * 3 + c]
                + fx  * fy   * row1[x1 * 3 + c];
        dst[c * dst_w * dst_h + idx] = v / 255.0f;
    }
}

// ─── Implementation ─────────────────────────────────────────────────────────

InferenceEngine::InferenceEngine(const PipelineConfig& config)
    : config_(config) {
    // Override input size from env if set
    if (auto* val = std::getenv("AVM_INFER_INPUT_SIZE")) {
        input_size_ = static_cast<uint32_t>(std::atoi(val));
    }
}

InferenceEngine::~InferenceEngine() {
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    // In production: delete TRT engine / context
}

bool InferenceEngine::loadEngine(cudaStream_t stream) {
    if (engine_loaded_) return true;

    // ─── Deserialize TensorRT engine ────────────────────────────────────
    // In production:
    //   1. Read file into memory: std::ifstream(config_.model_path, binary)
    //   2. nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    //   3. engine_ = runtime->deserializeCudaEngine(model_data, size);
    //   4. context_ = engine_->createExecutionContext();
    //
    std::cout << "[InferenceEngine] Loading model: " << config_.model_path << std::endl;

    // Allocate input/output buffers
    size_t input_elements = 3 * input_size_ * input_size_;  // NCHW
    size_t output_elements = constants::MAX_DETECTIONS * 7;    // [x, y, w, h, conf, class_id, batch_id]

    CUDA_CHECK(cudaMalloc(&d_input_, input_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, output_elements * sizeof(float)));
    // Zero the output so the scaffold never reports garbage as detections.
    CUDA_CHECK(cudaMemset(d_output_, 0, output_elements * sizeof(float)));
    output_size_ = output_elements;
    host_output_.resize(output_elements, 0.0f);

    engine_loaded_ = true;
    std::cout << "[InferenceEngine] Engine loaded successfully." << std::endl;

    (void)stream;
    return true;
}

bool InferenceEngine::preprocess(const CameraFrame& input, cudaStream_t stream) {
    if (!d_input_) return false;

    float scale_x = static_cast<float>(input.spec.width) / input_size_;
    float scale_y = static_cast<float>(input.spec.height) / input_size_;

    dim3 block(32, 8);
    dim3 grid((input_size_ + block.x - 1) / block.x,
              (input_size_ + block.y - 1) / block.y);

    kernel_preprocess_rgb8_to_float<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(input.device_data),
        static_cast<int>(input.device_pitch),
        static_cast<int>(input.spec.width),
        static_cast<int>(input.spec.height),
        static_cast<float*>(d_input_),
        static_cast<int>(input_size_),
        static_cast<int>(input_size_),
        scale_x, scale_y);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool InferenceEngine::inferAsync(const CameraFrame& input, cudaStream_t stream) {
    if (!engine_loaded_) return false;

    // 1. Preprocess input
    if (!preprocess(input, stream)) return false;

    // 2. Launch TensorRT inference (async on stream)
    // In production:
    //   context_->enqueueV2(buffers, stream, nullptr);
    //
    // For scaffold, simulate inference with a simple kernel.
    // We'll just record the event for async tracking.
    //

    infer_done_.record(stream);
    inference_launched_ = true;
    return true;
}

bool InferenceEngine::fetchResults(DetectionResult& out) {
    if (!inference_launched_) return false;

    // Synchronize (in production, check event first)
    infer_done_.sync();

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(host_output_.data(), d_output_,
                           output_size_ * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Parse detections
    out.boxes.clear();
    out.timestamp_us = 0;  // would be from frame metadata

    for (size_t i = 0; i < output_size_; i += 7) {
        float conf = host_output_[i + 4];
        if (conf < config_.detection_threshold) continue;

        BoundingBox box;
        box.x          = host_output_[i + 0];
        box.y          = host_output_[i + 1];
        box.w          = host_output_[i + 2];
        box.h          = host_output_[i + 3];
        box.confidence = conf;
        box.class_id   = static_cast<int>(host_output_[i + 5]);
        out.boxes.push_back(box);
    }

    inference_launched_ = false;
    return true;
}

bool InferenceEngine::isReady() const {
    if (!inference_launched_) return true;
    cudaError_t err = cudaEventQuery(infer_done_.get());
    return (err == cudaSuccess);
}

} // namespace avm
