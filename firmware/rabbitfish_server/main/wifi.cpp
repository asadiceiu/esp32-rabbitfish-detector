#include "wifi.hpp"

#include <cctype>
#include <cstring>
#include <string>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "mdns.h"
#include <vector>
#include "wifi_secrets.h"

static const char *TAG = "wifi";
static EventGroupHandle_t s_events;
static constexpr EventBits_t kConnected = BIT0;
static bool s_scanned;

static std::string normalised(const char *s)
{
    std::string out;
    for (; *s; s++) {
        if (!isspace((unsigned char)*s)) {
            out += (char)tolower((unsigned char)*s);
        }
    }
    return out;
}

// After "network not found": scan once and say whether the configured network is visible, and
// how. Only the configured network is reported, never the neighbours'.
static void report_scan()
{
    uint16_t n = 0;
    esp_wifi_scan_get_ap_num(&n);
    std::vector<wifi_ap_record_t> aps(n);
    esp_wifi_scan_get_ap_records(&n, aps.data());
    bool found = false;
    for (uint16_t i = 0; i < n; i++) {
        const char *ssid = (const char *)aps[i].ssid;
        bool exact = strcmp(ssid, WIFI_SSID) == 0;
        if (exact || normalised(ssid) == normalised(WIFI_SSID)) {
            found = true;
            ESP_LOGW(TAG, "scan: configured network %s on channel %d, RSSI %d dBm, security mode %d%s",
                     exact ? "found" : "found with different case/spaces", aps[i].primary, aps[i].rssi, aps[i].authmode,
                     aps[i].authmode < WIFI_AUTH_WPA2_PSK ? " (older than WPA2: not accepted)" : "");
        }
    }
    if (!found) {
        ESP_LOGW(TAG, "scan: %u networks visible on 2.4 GHz, none named like the configured one "
                      "(5 GHz-only network, out of range, or a typo in wifi_secrets.h)", n);
    }
    std::string seen;
    for (uint16_t i = 0; i < n; i++) {
        seen += " ch" + std::to_string(aps[i].primary) + "/" + std::to_string(aps[i].rssi) + "dBm";
    }
    ESP_LOGW(TAG, "scan: visible (channel/signal):%s", seen.c_str());
}

static void on_event(void *, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        auto *e = static_cast<wifi_event_sta_disconnected_t *>(data);
        ESP_LOGW(TAG, "disconnected (reason %d), retrying", e->reason);
        xEventGroupClearBits(s_events, kConnected);
        if (e->reason == WIFI_REASON_NO_AP_FOUND && !s_scanned) {
            s_scanned = true;
            esp_wifi_scan_start(nullptr, false); // reconnects on WIFI_EVENT_SCAN_DONE
            return;
        }
        vTaskDelay(pdMS_TO_TICKS(2000));
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_SCAN_DONE) {
        report_scan();
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto *e = static_cast<ip_event_got_ip_t *>(data);
        ESP_LOGI(TAG, "connected, IP " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_events, kConnected);
    }
}

esp_err_t wifi_start(const char *hostname, int timeout_ms)
{
    s_events = xEventGroupCreate();
    esp_netif_t *netif = esp_netif_create_default_wifi_sta();
    esp_netif_set_hostname(netif, hostname);

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, nullptr)); // incl. SCAN_DONE
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, nullptr));

    wifi_config_t cfg = {};
    strncpy((char *)cfg.sta.ssid, WIFI_SSID, sizeof(cfg.sta.ssid));
    strncpy((char *)cfg.sta.password, WIFI_PASSWORD, sizeof(cfg.sta.password));
    cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    cfg.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH; // home routers in WPA2/WPA3 mixed mode
    // Australia: channels 1-13 active. The default world-safe setting only listens passively on
    // 12-13, and a router there may never be found while connecting.
    ESP_ERROR_CHECK(esp_wifi_set_country_code("AU", true));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE); // power saving adds latency to every request
    ESP_LOGI(TAG, "joining \"%s\"", WIFI_SSID);

    ESP_ERROR_CHECK(mdns_init());
    mdns_hostname_set(hostname);
    mdns_instance_name_set("RabbitFish detector");
    mdns_service_add(nullptr, "_http", "_tcp", 80, nullptr, 0);

    EventBits_t bits = xEventGroupWaitBits(s_events, kConnected, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms));
    return (bits & kConnected) ? ESP_OK : ESP_ERR_TIMEOUT;
}
