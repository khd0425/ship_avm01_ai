#pragma once

#include "src/core/types.hpp"
#include "src/core/error_checks.hpp"
#include <memory>
#include <vector>
#include <string>

namespace avm {

// ─── InferenceEngine ────────────────────────────────────────────────────────
//
// Real-time maritime object detection using TensorRT FP16/INT8 quantized
// engine. Runs asynchronously on a dedicated CUDA stream to overlap
// inference with other pipeline stages.
//
// Supported classes (10~15 maritime objects):
//   vessel, buoy, net, floating_debris, person_in_water,
//   kayak, sailboat, navigation_mark, pier, bridge_pier, etc.
//
// Memory Flow:
//   Preprocessed image (device) → TensorRT context → detection output (host)
//
class InferenceEngine {
public:
    explicit InferenceEngine(const PipelineConfig& config);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = default;
    InferenceEngine& operator=(InferenceEngine&&) = default;

    /// Load and deserialize the TensorRT engine from file.
    /// @param stream  CUDA stream for async operations
    bool loadEngine(cudaStream_t stream);

    /// Run inference asynchronously.
    /// @param input   Preprocessed input image (device, RGB8, DETECTION_INPUT_SIZE²)
    /// @param stream  CUDA stream
    /// @return true if inference was launched successfully
    bool inferAsync(const CameraFrame& input, cudaStream_t stream);

    /// Synchronize and retrieve detections.
    /// @param out  Populated with detected objects
    /// @return true if detections are available
    bool fetchResults(DetectionResult& out);

    /// Check if inference is complete (non-blocking).
    bool isReady() const;

    /// Get the preprocessed input size.
    uint32_t inputSize() const { return input_size_; }

private:
    bool preprocess(const CameraFrame& input, cudaStream_t stream);

    PipelineConfig config_;
    uint32_t input_size_{640};

    // TensorRT handles (opaque pointers for scaffold)
    void* engine_{nullptr};
    void* context_{nullptr};

    // GPU buffers
    void* d_input_{nullptr};
    void* d_output_{nullptr};
    size_t output_size_{0};

    // Host output buffer
    std::vector<float> host_output_;

    // Async tracking
    CudaEvent infer_done_;
    bool engine_loaded_{false};
    bool inference_launched_{false};
};

} // namespace avm
