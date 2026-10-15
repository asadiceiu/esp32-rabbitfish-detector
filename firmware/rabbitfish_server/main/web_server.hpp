#pragma once

#include <string>
#include <vector>

#include "detector.hpp"
#include "esp_err.h"

// HTTP API and the built-in page, on port 80:
//   GET  /             the web page (index.html)
//   GET  /info         model description (model_info.json) plus self-test and runtime figures
//   GET  /health       {"ok":true}
//   POST /detect       body: a JPEG; query: conf (default 0.25), max_det (default 100)
//                      -> {"width","height","timing_ms":{...},"detections":[{"class","score","box"}]}
// Responses allow any origin (CORS), so a page hosted elsewhere can call the API too.
esp_err_t web_server_start(Detector *detector, const std::vector<std::string> *classes, const char *model_info_json);
