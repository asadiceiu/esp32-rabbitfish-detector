// RabbitFish detector server: loads the YOLO26n model, joins Wi-Fi, serves the API and web page.
// See web_server.hpp for the API and detector.hpp for what the model computes.
#include <string>
#include <vector>

#include "cJSON.h"
#include "detector.hpp"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "web_server.hpp"
#include "wifi.hpp"

static const char *TAG = "rabbitfish";
static const char *kHostname = "rabbitfish";

extern const char model_info_start[] asm("_binary_model_info_json_start");

static Detector s_detector;
static std::vector<std::string> s_classes;

extern "C" void app_main(void)
{
    esp_err_t err = nvs_flash_init(); // Wi-Fi keeps calibration data in NVS
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // model_info.json is written by tools/select_model.py together with model/model.espdl.
    cJSON *info = cJSON_Parse(model_info_start);
    std::vector<int> strides;
    cJSON *item;
    cJSON_ArrayForEach(item, cJSON_GetObjectItem(info, "strides")) strides.push_back(item->valueint);
    cJSON_ArrayForEach(item, cJSON_GetObjectItem(info, "classes")) s_classes.push_back(item->valuestring);
    float nms_iou = (float)cJSON_GetObjectItem(info, "nms_iou")->valuedouble;
    ESP_LOGI(TAG, "model %s / %s", cJSON_GetObjectItem(info, "run")->valuestring,
             cJSON_GetObjectItem(info, "variant")->valuestring);
    cJSON_Delete(info);

    // Internal RAM order: the server task's 16 KB stack first, then Wi-Fi, then the model. The
    // w8a16 model alone leaves too little internal RAM for the Wi-Fi driver to start.
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_LOGI(TAG, "internal RAM before server: %u B free, largest block %u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    ESP_ERROR_CHECK(web_server_start(&s_detector, &s_classes, model_info_start));

    if (wifi_start(kHostname, 30000) == ESP_OK) {
        ESP_LOGI(TAG, "ready: http://%s.local/", kHostname);
    } else {
        ESP_LOGW(TAG, "no Wi-Fi yet; still trying in the background (check main/wifi_secrets.h)");
    }
    ESP_LOGI(TAG, "internal RAM after Wi-Fi: %u B free, largest block %u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    ESP_ERROR_CHECK(s_detector.init(strides, nms_iou));
    ESP_LOGI(TAG, "internal RAM after model: %u B free, largest block %u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}
