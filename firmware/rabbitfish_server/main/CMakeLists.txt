if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/wifi_secrets.h)
    message(FATAL_ERROR "main/wifi_secrets.h is missing: copy main/wifi_secrets.h.example and fill in your Wi-Fi name and password.")
endif()
set(model_file ${CMAKE_CURRENT_LIST_DIR}/../model/model.espdl)
set(model_info ${CMAKE_CURRENT_LIST_DIR}/../model/model_info.json)
if(NOT EXISTS ${model_file} OR NOT EXISTS ${model_info})
    message(FATAL_ERROR "model/model.espdl or model/model_info.json is missing: run tools/select_model.py.")
endif()

idf_component_register(SRCS app_main.cpp detector.cpp web_server.cpp wifi.cpp
                       INCLUDE_DIRS "."
                       REQUIRES esp-dl esp_http_server esp_wifi esp_netif esp_event nvs_flash esp_timer json
                       EMBED_TXTFILES index.html ${model_info})

# The model goes into its own flash partition, written by `idf.py flash`.
esptool_py_flash_to_partition(flash "model" "${model_file}")
