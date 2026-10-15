#include "web_server.hpp"

#include <cstdio>
#include <cstdlib>

#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "web";
static constexpr size_t kMaxUpload = 3 * 1024 * 1024;

extern const char index_html_start[] asm("_binary_index_html_start");
extern const char index_html_end[] asm("_binary_index_html_end");

static Detector *s_detector;
static const std::vector<std::string> *s_classes;
static const char *s_model_info;
static uint32_t s_requests;

static void cors(httpd_req_t *req)
{
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type");
}

static esp_err_t send_json(httpd_req_t *req, const char *status, const std::string &body)
{
    cors(req);
    httpd_resp_set_status(req, status);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, body.data(), body.size());
}

static esp_err_t send_error(httpd_req_t *req, const char *status, const std::string &message)
{
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "error", message.c_str());
    char *s = cJSON_PrintUnformatted(o);
    esp_err_t r = send_json(req, status, s);
    cJSON_free(s);
    cJSON_Delete(o);
    return r;
}

static esp_err_t index_get(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html_start, index_html_end - index_html_start - 1); // drop EMBED_TXTFILES' NUL
}

static esp_err_t info_get(httpd_req_t *req)
{
    cJSON *o = cJSON_Parse(s_model_info);
    cJSON *rt = cJSON_AddObjectToObject(o, "device");
    cJSON_AddBoolToObject(rt, "self_test_passed", s_detector->self_test_passed());
    cJSON_AddNumberToObject(rt, "boot_inference_ms", s_detector->boot_inference_us() / 1000);
    cJSON_AddStringToObject(rt, "runtime", s_detector->dual_core() ? "dual-core" : "single-core");
    cJSON_AddNumberToObject(rt, "uptime_s", esp_timer_get_time() / 1000000);
    cJSON_AddNumberToObject(rt, "requests", s_requests);
    cJSON_AddNumberToObject(rt, "free_internal_bytes", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    cJSON_AddNumberToObject(rt, "free_psram_bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    char *s = cJSON_PrintUnformatted(o);
    esp_err_t r = send_json(req, "200 OK", s);
    cJSON_free(s);
    cJSON_Delete(o);
    return r;
}

static esp_err_t health_get(httpd_req_t *req)
{
    return send_json(req, "200 OK", "{\"ok\":true}");
}

static esp_err_t options(httpd_req_t *req)
{
    cors(req);
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, nullptr, 0);
}

static float query_float(httpd_req_t *req, const char *key, float fallback)
{
    char query[128], value[32];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, key, value, sizeof(value)) == ESP_OK) {
        return strtof(value, nullptr);
    }
    return fallback;
}

static esp_err_t detect_post(httpd_req_t *req)
{
    int64_t t_start = esp_timer_get_time();
    if (req->content_len == 0 || req->content_len > kMaxUpload) {
        return send_error(req, "413 Payload Too Large", "send a JPEG of at most 3 MB as the request body");
    }
    uint8_t *buf = static_cast<uint8_t *>(heap_caps_malloc(req->content_len, MALLOC_CAP_SPIRAM));
    if (!buf) {
        return send_error(req, "503 Service Unavailable", "out of memory");
    }
    size_t got = 0;
    while (got < req->content_len) {
        int n = httpd_req_recv(req, (char *)buf + got, req->content_len - got);
        if (n == HTTPD_SOCK_ERR_TIMEOUT) {
            continue;
        }
        if (n <= 0) {
            heap_caps_free(buf);
            return ESP_FAIL; // connection dropped
        }
        got += n;
    }
    int64_t t_received = esp_timer_get_time();

    float conf = query_float(req, "conf", 0.25f);
    int max_det = (int)query_float(req, "max_det", 100);
    std::vector<Detection> dets;
    int w = 0, h = 0;
    DetectTiming t;
    std::string error;
    esp_err_t err = s_detector->detect_jpeg(buf, got, conf, max_det, dets, w, h, t, error);
    heap_caps_free(buf);
    if (err != ESP_OK) {
        return send_error(req, "400 Bad Request", error);
    }
    s_requests++;

    std::string out;
    out.reserve(256 + dets.size() * 96);
    char tmp[200];
    snprintf(tmp, sizeof(tmp),
             "{\"width\":%d,\"height\":%d,\"timing_ms\":{\"receive\":%.1f,\"decode\":%.1f,\"preprocess\":%.1f,"
             "\"inference\":%.1f,\"postprocess\":%.1f},\"detections\":[",
             w, h, (t_received - t_start) / 1000.0, t.decode_us / 1000.0, t.preprocess_us / 1000.0,
             t.inference_us / 1000.0, t.postprocess_us / 1000.0);
    out += tmp;
    for (size_t i = 0; i < dets.size(); i++) {
        const Detection &d = dets[i];
        snprintf(tmp, sizeof(tmp), "%s{\"class\":\"%s\",\"class_id\":%d,\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
                 i ? "," : "", (*s_classes)[d.cls].c_str(), d.cls, d.score, d.x1, d.y1, d.x2, d.y2);
        out += tmp;
    }
    out += "]}";
    ESP_LOGI(TAG, "detect %dx%d, %u B: %d boxes; decode %.0f, pre %.0f, infer %.0f, post %.0f ms", w, h,
             (unsigned)got, (int)dets.size(), t.decode_us / 1000.0, t.preprocess_us / 1000.0,
             t.inference_us / 1000.0, t.postprocess_us / 1000.0);
    return send_json(req, "200 OK", out);
}

esp_err_t web_server_start(Detector *detector, const std::vector<std::string> *classes, const char *model_info_json)
{
    s_detector = detector;
    s_classes = classes;
    s_model_info = model_info_json;

    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 16384; // inference runs in the server task
    cfg.lru_purge_enable = true;
    cfg.recv_wait_timeout = 20;
    cfg.send_wait_timeout = 20;
    httpd_handle_t server = nullptr;
    esp_err_t err = httpd_start(&server, &cfg);
    if (err != ESP_OK) {
        return err;
    }
    const httpd_uri_t uris[] = {
        {"/", HTTP_GET, index_get, nullptr},
        {"/info", HTTP_GET, info_get, nullptr},
        {"/health", HTTP_GET, health_get, nullptr},
        {"/detect", HTTP_POST, detect_post, nullptr},
        {"/detect", HTTP_OPTIONS, options, nullptr},
    };
    for (const auto &u : uris) {
        httpd_register_uri_handler(server, &u);
    }
    ESP_LOGI(TAG, "HTTP server on port %d", cfg.server_port);
    return ESP_OK;
}
