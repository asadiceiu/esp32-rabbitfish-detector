# RabbitFish detector as an HTTP inference server on an ESP32-S3 (N16R8).
# Build: activate ESP-IDF (. $IDF_PATH/export.sh), then idf.py set-target esp32s3 && idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(rabbitfish_server)
