// RabbitFish YOLO26 detector on ESP-DL: JPEG in, boxes out.
//
// Mirrors esp32_export.py (repo root), which produced the model and measured its mAP:
//   preprocessing  letterbox the RGB image into the model input (keep aspect, pad 114, /255)
//   outputs        per stride s: box_s<s> (1,h,w,4) = l,t,r,b distances in cells,
//                                cls_s<s> (1,h,w,nc) = class logits (NHWC in ESP-DL)
//   decoding       x1=(gx+0.5-l)*s, y1=(gy+0.5-t)*s, x2=(gx+0.5+r)*s, y2=(gy+0.5+b)*s,
//                  score=sigmoid(logit), then per-class NMS
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "esp_err.h"

namespace dl {
class Model;
class TensorBase;
}

struct Detection {
    int cls;
    float score;
    float x1, y1, x2, y2; // pixels of the uploaded image
};

struct DetectTiming {
    int64_t decode_us = 0, preprocess_us = 0, inference_us = 0, postprocess_us = 0;
};

class Detector {
public:
    // Loads the model from the "model" flash partition and runs the self-test embedded in the
    // .espdl (one val image and the outputs esp-ppq's simulator computed for it).
    esp_err_t init(const std::vector<int> &strides, float nms_iou);

    // Decodes a JPEG and detects. On failure returns an error and sets `error` to a message.
    esp_err_t detect_jpeg(const uint8_t *jpeg, size_t len, float conf, int max_det, std::vector<Detection> &out,
                          int &img_w, int &img_h, DetectTiming &timing, std::string &error);

    int input_width() const { return in_w_; }
    int input_height() const { return in_h_; }
    bool self_test_passed() const { return self_test_ok_; }
    int64_t boot_inference_us() const { return boot_inference_us_; }
    bool dual_core() const { return dual_core_; }

private:
    void preprocess(const uint8_t *rgb, int w, int h, float &r, int &px, int &py);
    template <typename T>
    void letterbox(T *dst, const uint8_t *rgb, int w, int h, int nw, int nh, int px, int py);
    void postprocess(float conf, int max_det, std::vector<Detection> &out);

    dl::Model *model_ = nullptr;
    dl::TensorBase *input_ = nullptr;
    std::vector<int> strides_;
    std::vector<dl::TensorBase *> box_out_, cls_out_;
    float nms_iou_ = 0.7f;
    int in_w_ = 0, in_h_ = 0;
    int16_t lut_[256]; // pixel value -> quantised model input (stored as int8 or int16, per the model)
    bool input16_ = false;
    std::atomic<bool> ready_{false}; // init() finished; the server may start before it does
    bool self_test_ok_ = false;
    int64_t boot_inference_us_ = 0;
    bool dual_core_ = true; // the faster runtime mode, measured at boot
};
