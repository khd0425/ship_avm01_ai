#pragma once

// AI object detector on OpenCV DNN (YOLOX ONNX, Apache-2.0) for machines without
// TensorRT - the development PC. On the Jetson the same interface is served by
// TensorRT (InferenceEngine). Runs in its own thread so the video pipeline never
// waits for it (spec: AI inference >= 15 fps is a separate requirement, PR-5).
//
// Model: YOLOX-s exported to ONNX, output [1, 8400, 85] = (cx, cy, w, h, obj, 80 COCO classes),
// undecoded. Only COCO person (0) and boat (8) are used, mapped to the AVM classes
// "person" and "small_vessel". The remaining AVM classes need a model trained on
// maritime data (docs/ai_dataset_plan.md).

#ifdef AVM_WITH_OPENCV

#include "src/core/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace avm {

class DnnDetector {
public:
    struct Config {
        std::string model_path;
        int input_size{640};
        float confidence{0.4f};
        float nms{0.45f};
        std::vector<std::string> classes;   // AVM class names; index = BoundingBox::class_id
        /// "yolox": COCO-pretrained YOLOX (only person/boat are used, mapped to AVM classes).
        /// "yolov8": model trained on the AVM classes (ai/train.py); output [1, 4+N, 8400], class id = AVM id.
        std::string arch{"yolox"};
    };

    DnnDetector() = default;
    ~DnnDetector() { stop(); }
    DnnDetector(const DnnDetector&) = delete;
    DnnDetector& operator=(const DnnDetector&) = delete;

    bool start(const Config& cfg, std::string* err = nullptr);
    void stop();

    /// True if the worker can take a new image.
    bool idle() const { return !busy_.load(); }

    /// Hand an RGB8 image to the worker (copied). False if the worker is still busy.
    bool submit(const std::uint8_t* rgb, std::size_t pitch, int width, int height);

    /// Latest finished detections (boxes normalised to the submitted image, top-left origin).
    /// Returns false if nothing new since the last call.
    bool fetch(DetectionResult& out);

    double last_inference_ms() const { return last_ms_.load(); }
    const std::vector<std::string>& classes() const { return cfg_.classes; }

    /// Pure function (unit-testable): decode YOLOX raw output into boxes in the
    /// letterboxed input space, keeping only the listed COCO classes.
    struct RawBox { float x, y, w, h, score; int coco_class; };
    /// YOLOv8 head: data laid out [4 + num_classes][num_anchors] (channel-major), boxes as
    /// centre/size in input pixels, no objectness. Anchors with score < confidence are dropped.
    static std::vector<RawBox> decode_yolov8(const float* data, int channels, int anchors,
                                             float confidence);

    static std::vector<RawBox> decode_yolox(const float* data, int num, int dim, int input_size,
                                            float confidence, const std::vector<int>& coco_keep);

private:
    void worker();
    std::vector<BoundingBox> detect(const cv::Mat& bgr);

    Config cfg_;
    cv::dnn::Net net_;
    int person_id_{-1}, vessel_id_{-1};

    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    cv::Mat job_;
    bool have_job_{false};
    bool quit_{false};
    std::atomic<bool> busy_{false};

    DetectionResult result_;
    bool result_new_{false};
    std::atomic<double> last_ms_{0.0};
    std::uint32_t frame_counter_{0};
};

} // namespace avm

#endif
