"""RabbitFish detector: YOLO26 training, ESP32-S3 quantisation and evaluation helpers.

Modules
    paths    where the dataset, runs and models live (RABBITFISH_DATASET overrides the dataset)
    data     dataset listing, labels and the letterbox preprocessing shared with the firmware
    metrics  box decoding (mirrors the firmware), ultralytics-style matching and mAP
    export   ONNX export of the raw YOLO26 head and esp-ppq quantisation for the ESP32-S3
"""
