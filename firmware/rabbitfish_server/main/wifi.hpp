#pragma once

#include "esp_err.h"

// Joins the Wi-Fi network in wifi_secrets.h (retrying forever in the background) and announces
// the device as http://<hostname>.local via mDNS. Returns once connected or after timeout_ms.
// esp_netif_init() and esp_event_loop_create_default() must have been called.
esp_err_t wifi_start(const char *hostname, int timeout_ms);
