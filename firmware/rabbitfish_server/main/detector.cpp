#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "dl_image_jpeg.hpp"
#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "detector";

// Decoded RGB888 images above this many pixels are refused: a 1920x1440 photo alone would need
// 8.3 MB, more than the PSRAM. The web page and the client resize to the model input first.
static constexpr int kMaxPixels = 1280 * 1024;
static constexpr uint8_t kPad = 114; // letterbox fill, as in training

// Width and height from the JPEG's start-of-frame marker, without decoding.
static bool jpeg_size(const uint8_t *p, size_t len, int &w, int &h)
{
    if (len < 4 || p[0] != 0xFF || p[1] != 0xD8) {
        return false;
    }
    size_t i = 2;
    while (i + 9 < len) {
        if (p[i] != 0xFF) {
            return false;
        }
        uint8_t marker = p[i + 1];
        if (marker == 0xFF) { // fill byte
            i++;
            continue;
        }
        size_t seg = (p[i + 2] << 8) | p[i + 3];
        bool sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (sof) {
            h = (p[i + 5] << 8) | p[i + 6];
            w = (p[i + 7] << 8) | p[i + 8];
            return w > 0 && h > 0;
        }
        i += 2 + seg;
    }
    return false;
}

static float value_at(const dl::TensorBase *t, int idx, float scale)
{
    return t->dtype == dl::DATA_TYPE_INT16 ? static_cast<const int16_t *>(t->data)[idx] * scale
                                           : static_cast<const int8_t *>(t->data)[idx] * scale;
}

esp_err_t Detector::init(const std::vector<int> &strides, float nms_iou)
{
    strides_ = strides;
    nms_iou_ = nms_iou;

    // param_copy=true: weights are copied from flash to PSRAM, which is faster to read.
    model_ = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION, 0, dl::MEMORY_MANAGER_GREEDY, nullptr,
                           true);
    if (!model_) {
        return ESP_FAIL;
    }
    self_test_ok_ = model_->test() == ESP_OK;
    ESP_LOGI(TAG, "self-test against esp-ppq's expected outputs: %s", self_test_ok_ ? "PASS" : "FAIL");

    input_ = model_->get_input();
    const auto &s = input_->get_shape(); // NHWC
    if (s.size() != 4 || s[3] != 3 || (input_->dtype != dl::DATA_TYPE_INT8 && input_->dtype != dl::DATA_TYPE_INT16)) {
        ESP_LOGE(TAG, "unexpected model input: %d dims, dtype %s", (int)s.size(), input_->get_dtype_string());
        return ESP_FAIL;
    }
    in_h_ = s[1];
    in_w_ = s[2];
    input16_ = input_->dtype == dl::DATA_TYPE_INT16; // w8a16 / w16a16 exports
    int qmax = input16_ ? 32767 : 127;
    float q = 1.0f / DL_SCALE(int(input_->exponent)); // input = round(pixel / 255 * q)
    for (int v = 0; v < 256; v++) {
        lut_[v] = (int16_t)std::clamp((int)std::lround(v / 255.0f * q), -qmax - 1, qmax);
    }
    ESP_LOGI(TAG, "input %s, exponent %d", input_->get_dtype_string(), int(input_->exponent));

    for (int st : strides_) {
        dl::TensorBase *b = model_->get_output("box_s" + std::to_string(st));
        dl::TensorBase *c = model_->get_output("cls_s" + std::to_string(st));
        if (!b || !c) {
            ESP_LOGE(TAG, "model has no box_s%d / cls_s%d output", st, st);
            return ESP_FAIL;
        }
        if (b->get_size() != b->shape[1] * b->shape[2] * 4 || c->get_size() != c->shape[1] * c->shape[2] * c->shape[3]) {
            ESP_LOGE(TAG, "stride %d outputs are padded; the decoder assumes dense NHWC", st);
            return ESP_FAIL;
        }
        box_out_.push_back(b);
        cls_out_.push_back(c);
        ESP_LOGI(TAG, "stride %2d: box %dx%dx%d %s exp %d, cls %dx%dx%d %s exp %d", st, b->shape[1], b->shape[2],
                 b->shape[3], b->get_dtype_string(), int(b->exponent), c->shape[1], c->shape[2], c->shape[3],
                 c->get_dtype_string(), int(c->exponent));
    }

    // Time one inference on a blank (padding) image, single core and dual core.
    if (input16_) {
        std::fill_n(static_cast<int16_t *>(input_->data), input_->get_size(), lut_[kPad]);
    } else {
        std::fill_n(static_cast<int8_t *>(input_->data), input_->get_size(), (int8_t)lut_[kPad]);
    }
    int64_t t0 = esp_timer_get_time();
    model_->run(dl::RUNTIME_MODE_SINGLE_CORE);
    int64_t t1 = esp_timer_get_time();
    model_->run(dl::RUNTIME_MODE_MULTI_CORE);
    int64_t t2 = esp_timer_get_time();
    dual_core_ = t2 - t1 < t1 - t0;
    boot_inference_us_ = std::min(t1 - t0, t2 - t1);
    ESP_LOGI(TAG, "input %dx%d; inference %.0f ms single-core, %.0f ms dual-core; using %s", in_w_, in_h_,
             (t1 - t0) / 1000.0, (t2 - t1) / 1000.0, dual_core_ ? "dual-core" : "single-core");
    ESP_LOGI(TAG, "free heap: internal %u B, PSRAM %u B", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ready_ = true;
    return ESP_OK;
}

// Letterbox into the model input, quantising on the way: same geometry as esp32_export.py's
// letterbox() and bilinear sampling with cv2.INTER_LINEAR's half-pixel convention.
void Detector::preprocess(const uint8_t *rgb, int w, int h, float &r, int &px, int &py)
{
    r = std::min((float)in_w_ / w, (float)in_h_ / h);
    int nw = std::lround(w * r), nh = std::lround(h * r);
    px = (in_w_ - nw) / 2;
    py = (in_h_ - nh) / 2;
    if (input16_) {
        letterbox(static_cast<int16_t *>(input_->data), rgb, w, h, nw, nh, px, py);
    } else {
        letterbox(static_cast<int8_t *>(input_->data), rgb, w, h, nw, nh, px, py);
    }
}

template <typename T>
void Detector::letterbox(T *dst, const uint8_t *rgb, int w, int h, int nw, int nh, int px, int py)
{
    std::fill_n(dst, (size_t)in_w_ * in_h_ * 3, (T)lut_[kPad]);

    if (nw == w && nh == h) { // already the right size, e.g. 320x240 from the web page
        for (int y = 0; y < h; y++) {
            const uint8_t *s = rgb + (size_t)y * w * 3;
            T *d = dst + ((size_t)(y + py) * in_w_ + px) * 3;
            for (int i = 0; i < w * 3; i++) {
                d[i] = (T)lut_[s[i]];
            }
        }
        return;
    }
    float sx = (float)w / nw, sy = (float)h / nh;
    for (int y = 0; y < nh; y++) {
        float fy = std::max((y + 0.5f) * sy - 0.5f, 0.0f);
        int y0 = std::min((int)fy, h - 1), y1 = std::min(y0 + 1, h - 1);
        float wy = fy - y0;
        T *d = dst + ((size_t)(y + py) * in_w_ + px) * 3;
        for (int x = 0; x < nw; x++) {
            float fx = std::max((x + 0.5f) * sx - 0.5f, 0.0f);
            int x0 = std::min((int)fx, w - 1), x1 = std::min(x0 + 1, w - 1);
            float wx = fx - x0;
            const uint8_t *a = rgb + ((size_t)y0 * w + x0) * 3, *b = rgb + ((size_t)y0 * w + x1) * 3;
            const uint8_t *c = rgb + ((size_t)y1 * w + x0) * 3, *e = rgb + ((size_t)y1 * w + x1) * 3;
            for (int k = 0; k < 3; k++) {
                float top = a[k] + (b[k] - a[k]) * wx, bot = c[k] + (e[k] - c[k]) * wx;
                d[x * 3 + k] = (T)lut_[(int)std::lround(top + (bot - top) * wy)];
            }
        }
    }
}

static float iou(const Detection &a, const Detection &b)
{
    float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (iw <= 0 || ih <= 0) {
        return 0;
    }
    float inter = iw * ih;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// Boxes in model-input pixels, score above conf, per-class NMS, best max_det.
void Detector::postprocess(float conf, int max_det, std::vector<Detection> &out)
{
    conf = std::clamp(conf, 1e-4f, 0.9999f);
    float logit_thr = std::log(conf / (1 - conf)); // compare logits, sigmoid only for kept cells
    std::vector<Detection> cand;
    for (size_t i = 0; i < strides_.size(); i++) {
        const dl::TensorBase *b = box_out_[i], *c = cls_out_[i];
        int h = c->shape[1], w = c->shape[2], nc = c->shape[3];
        float bs = DL_SCALE(int(b->exponent)), cs = DL_SCALE(int(c->exponent)), s = strides_[i];
        for (int gy = 0; gy < h; gy++) {
            for (int gx = 0; gx < w; gx++) {
                int cell = gy * w + gx;
                for (int k = 0; k < nc; k++) {
                    float logit = value_at(c, cell * nc + k, cs);
                    if (logit <= logit_thr) {
                        continue;
                    }
                    float l = value_at(b, cell * 4 + 0, bs), t = value_at(b, cell * 4 + 1, bs);
                    float r = value_at(b, cell * 4 + 2, bs), bt = value_at(b, cell * 4 + 3, bs);
                    cand.push_back({k, 1.0f / (1.0f + std::exp(-logit)), (gx + 0.5f - l) * s, (gy + 0.5f - t) * s,
                                    (gx + 0.5f + r) * s, (gy + 0.5f + bt) * s});
                }
            }
        }
    }
    std::sort(cand.begin(), cand.end(), [](const Detection &a, const Detection &b) { return a.score > b.score; });
    out.clear();
    std::vector<bool> removed(cand.size(), false);
    for (size_t i = 0; i < cand.size() && (int)out.size() < max_det; i++) {
        if (removed[i]) {
            continue;
        }
        out.push_back(cand[i]);
        for (size_t j = i + 1; j < cand.size(); j++) {
            if (!removed[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > nms_iou_) {
                removed[j] = true;
            }
        }
    }
}

esp_err_t Detector::detect_jpeg(const uint8_t *jpeg, size_t len, float conf, int max_det,
                                std::vector<Detection> &out, int &img_w, int &img_h, DetectTiming &timing,
                                std::string &error)
{
    if (!ready_) {
        error = "the model is still loading; try again in a few seconds";
        return ESP_ERR_INVALID_STATE;
    }
    if (!jpeg_size(jpeg, len, img_w, img_h)) {
        error = "not a JPEG image";
        return ESP_ERR_INVALID_ARG;
    }
    if (img_w * img_h > kMaxPixels) {
        error = "image is " + std::to_string(img_w) + "x" + std::to_string(img_h) +
            "; resize it to at most 1280x1024 before uploading (the model input is " + std::to_string(in_w_) + "x" +
            std::to_string(in_h_) + ")";
        return ESP_ERR_INVALID_SIZE;
    }

    int64_t t0 = esp_timer_get_time();
    dl::image::img_t img = dl::image::sw_decode_jpeg({(void *)jpeg, len}, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    if (!img.data) {
        error = "JPEG decoding failed (progressive JPEGs are not supported)";
        return ESP_FAIL;
    }
    int64_t t1 = esp_timer_get_time();
    float r;
    int px, py;
    preprocess(static_cast<const uint8_t *>(img.data), img.width, img.height, r, px, py);
    heap_caps_free(img.data);
    int64_t t2 = esp_timer_get_time();
    model_->run(dual_core_ ? dl::RUNTIME_MODE_MULTI_CORE : dl::RUNTIME_MODE_SINGLE_CORE);
    int64_t t3 = esp_timer_get_time();
    postprocess(conf, max_det, out);
    for (auto &d : out) { // model-input pixels -> uploaded-image pixels
        d.x1 = std::clamp((d.x1 - px) / r, 0.0f, (float)img_w);
        d.x2 = std::clamp((d.x2 - px) / r, 0.0f, (float)img_w);
        d.y1 = std::clamp((d.y1 - py) / r, 0.0f, (float)img_h);
        d.y2 = std::clamp((d.y2 - py) / r, 0.0f, (float)img_h);
    }
    int64_t t4 = esp_timer_get_time();
    timing = {t1 - t0, t2 - t1, t3 - t2, t4 - t3};
    return ESP_OK;
}
