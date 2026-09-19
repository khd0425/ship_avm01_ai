#ifdef AVM_WITH_OPENCV

#include "dnn_detector.hpp"

#include "src/util/logger.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace avm {

namespace {
constexpr int kCocoPerson = 0;
constexpr int kCocoBoat = 8;
}

std::vector<DnnDetector::RawBox> DnnDetector::decode_yolox(const float* p, int num, int dim, int input_size,
                                                          float confidence, const std::vector<int>& coco_keep) {
    std::vector<RawBox> out;
    if (dim < 6) return out;
    const int strides[3] = {8, 16, 32};
    int index = 0;
    for (int s : strides) {
        const int g = input_size / s;
        for (int gy = 0; gy < g; ++gy) {
            for (int gx = 0; gx < g; ++gx, ++index) {
                if (index >= num) return out;
                const float* r = p + static_cast<std::size_t>(index) * dim;
                const float obj = r[4];
                if (obj < confidence) continue;                     // objectness is a hard upper bound of the score
                int best = -1;
                float best_cls = 0.0f;
                for (int c : coco_keep) {
                    if (5 + c < dim && r[5 + c] > best_cls) { best_cls = r[5 + c]; best = c; }
                }
                const float score = obj * best_cls;
                if (best < 0 || score < confidence) continue;
                const float cx = (r[0] + static_cast<float>(gx)) * static_cast<float>(s);
                const float cy = (r[1] + static_cast<float>(gy)) * static_cast<float>(s);
                const float w = std::exp(r[2]) * static_cast<float>(s);
                const float h = std::exp(r[3]) * static_cast<float>(s);
                out.push_back({cx - 0.5f * w, cy - 0.5f * h, w, h, score, best});
            }
        }
    }
    return out;
}

std::vector<DnnDetector::RawBox> DnnDetector::decode_yolov8(const float* p, int channels, int anchors,
                                                            float confidence) {
    std::vector<RawBox> out;
    const int nc = channels - 4;
    if (nc <= 0) return out;
    for (int a = 0; a < anchors; ++a) {
        int best = -1;
        float best_s = confidence;
        for (int c = 0; c < nc; ++c) {
            const float s = p[static_cast<std::size_t>(4 + c) * anchors + a];
            if (s >= best_s) { best_s = s; best = c; }
        }
        if (best < 0) continue;
        const float cx = p[a], cy = p[static_cast<std::size_t>(anchors) + a];
        const float w = p[2 * static_cast<std::size_t>(anchors) + a], h = p[3 * static_cast<std::size_t>(anchors) + a];
        out.push_back({cx - 0.5f * w, cy - 0.5f * h, w, h, best_s, best});
    }
    return out;
}

bool DnnDetector::start(const Config& cfg, std::string* err) {
    stop();
    cfg_ = cfg;
    try {
        net_ = cv::dnn::readNetFromONNX(cfg.model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        if (err) *err = "cannot load ONNX model '" + cfg.model_path + "': " + e.what();
        return false;
    }
    auto find = [&](const char* name) {
        for (std::size_t i = 0; i < cfg_.classes.size(); ++i) if (cfg_.classes[i] == name) return static_cast<int>(i);
        return -1;
    };
    person_id_ = find("person");
    vessel_id_ = find("small_vessel");
    if (cfg.arch != "yolox" && cfg.arch != "yolov8") {
        if (err) *err = "inference arch must be 'yolox' or 'yolov8'";
        return false;
    }
    if (cfg.arch == "yolox" && person_id_ < 0 && vessel_id_ < 0) {
        if (err) *err = "class list has neither 'person' nor 'small_vessel'";
        return false;
    }
    quit_ = false;
    thread_ = std::thread([this] { worker(); });
    AVM_LOGI("detector", "%s (OpenCV DNN, CPU) loaded: %s | person->%d small_vessel->%d", cfg.arch.c_str(),
             cfg.model_path.c_str(), person_id_, vessel_id_);
    return true;
}

void DnnDetector::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool DnnDetector::submit(const std::uint8_t* rgb, std::size_t pitch, int width, int height) {
    if (busy_.exchange(true)) return false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cv::Mat src(height, width, CV_8UC3, const_cast<std::uint8_t*>(rgb), pitch);
        cv::cvtColor(src, job_, cv::COLOR_RGB2BGR);      // YOLOX expects BGR
        have_job_ = true;
    }
    cv_.notify_one();
    return true;
}

bool DnnDetector::fetch(DetectionResult& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!result_new_) return false;
    out = result_;
    result_new_ = false;
    return true;
}

void DnnDetector::worker() {
    while (true) {
        cv::Mat img;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return have_job_ || quit_; });
            if (quit_) return;
            img = std::move(job_);
            have_job_ = false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        DetectionResult res;
        try {
            res.boxes = detect(img);
        } catch (const cv::Exception& e) {
            AVM_LOGE("detector", "inference failed: %s", e.what());
        }
        last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(mu_);
            res.frame_id = ++frame_counter_;
            result_ = std::move(res);
            result_new_ = true;
        }
        busy_ = false;
    }
}

std::vector<BoundingBox> DnnDetector::detect(const cv::Mat& bgr) {
    const int S = cfg_.input_size;
    // Letterbox: keep aspect ratio, pad bottom/right with 114 (YOLOX convention).
    const float scale = std::min(static_cast<float>(S) / bgr.cols, static_cast<float>(S) / bgr.rows);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::Mat canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));

    const bool v8 = (cfg_.arch == "yolov8");
    // YOLOX: BGR 0..255.  YOLOv8: RGB 0..1.
    cv::Mat blob = v8 ? cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(S, S), cv::Scalar(), /*swapRB=*/true, false)
                      : cv::dnn::blobFromImage(canvas, 1.0, cv::Size(S, S), cv::Scalar(), /*swapRB=*/false, false);
    net_.setInput(blob);
    cv::Mat out = net_.forward();                         // yolox [1, 8400, 85] | yolov8 [1, 4+N, 8400]
    if (out.dims != 3) return {};

    std::vector<RawBox> raw;
    if (v8) {
        raw = decode_yolov8(reinterpret_cast<const float*>(out.data), out.size[1], out.size[2], cfg_.confidence);
    } else {
        std::vector<int> keep;
        if (person_id_ >= 0) keep.push_back(kCocoPerson);
        if (vessel_id_ >= 0) keep.push_back(kCocoBoat);
        raw = decode_yolox(reinterpret_cast<const float*>(out.data), out.size[1], out.size[2], S,
                           cfg_.confidence, keep);
    }

    std::vector<cv::Rect> rects;
    std::vector<float> scores;
    std::vector<int> cls;
    for (const auto& b : raw) {
        rects.emplace_back(static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.w), static_cast<int>(b.h));
        scores.push_back(b.score);
        cls.push_back(b.coco_class);
    }
    std::vector<int> keep_idx;
    cv::dnn::NMSBoxesBatched(rects, scores, cls, cfg_.confidence, cfg_.nms, keep_idx);   // per-class NMS

    std::vector<BoundingBox> boxes;
    for (int i : keep_idx) {
        const float x0 = std::max(0.0f, raw[static_cast<std::size_t>(i)].x / scale);
        const float y0 = std::max(0.0f, raw[static_cast<std::size_t>(i)].y / scale);
        const float x1 = std::min(static_cast<float>(bgr.cols), (raw[static_cast<std::size_t>(i)].x + raw[static_cast<std::size_t>(i)].w) / scale);
        const float y1 = std::min(static_cast<float>(bgr.rows), (raw[static_cast<std::size_t>(i)].y + raw[static_cast<std::size_t>(i)].h) / scale);
        if (x1 <= x0 || y1 <= y0) continue;
        BoundingBox bb;
        bb.x = x0 / bgr.cols;
        bb.y = y0 / bgr.rows;
        bb.w = (x1 - x0) / bgr.cols;
        bb.h = (y1 - y0) / bgr.rows;
        bb.confidence = scores[static_cast<std::size_t>(i)];
        bb.class_id = v8 ? cls[static_cast<std::size_t>(i)]
                         : ((cls[static_cast<std::size_t>(i)] == kCocoPerson) ? person_id_ : vessel_id_);
        boxes.push_back(bb);
    }
    return boxes;
}

} // namespace avm

#endif
