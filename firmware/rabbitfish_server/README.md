# RabbitFish detector server (ESP32-S3)

The quantised YOLO26n fish body/head detector running on an **ESP32-S3-N16R8** (16 MB flash,
8 MB PSRAM) as a Wi-Fi inference server. No camera: you upload a photo from any browser on the
same network and get the boxes back. Open `http://rabbitfish.local/` for the built-in page, or
call the JSON API.

- One inference: **~950 ms** (both cores); a full request about 1.07 s.
- Accuracy on the device, measured through this API: test mAP50 0.947, mAP50-95 0.643.
- Photos are only held in RAM while they are processed; nothing is written to flash.

## API

| Request | Returns |
|---|---|
| `GET /` | the web page (drag in photos, boxes drawn in the browser) |
| `GET /info` | model, input size, classes, simulated val mAP, self-test result, boot inference time, free memory |
| `GET /health` | `{"ok":true}` |
| `POST /detect?conf=0.25&max_det=100` | body: a JPEG (at most 1280x1024; shrink to 320x240 for speed). Returns `{"width","height","timing_ms":{receive,decode,preprocess,inference,postprocess},"detections":[{"class","class_id","score","box":[x1,y1,x2,y2]}]}` with boxes in the uploaded image's pixels |

One request runs at a time (one model in memory). Any origin may call the API (CORS).

```bash
curl -s --data-binary @photo_320x240.jpg -H 'Content-Type: image/jpeg' 'http://rabbitfish.local/detect?conf=0.25'
```

## Build and flash

1. Install [ESP-IDF v5.5](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/get-started/)
   (the S3 tools are enough: `./install.sh esp32s3`) and activate it: `. $IDF_PATH/export.sh`.
2. Wi-Fi: `cp main/wifi_secrets.h.example main/wifi_secrets.h` and fill in your network. It must
   be a 2.4 GHz network (the S3 has no 5 GHz) and the name must match exactly. The file is git-ignored.
3. Model: `model/` already holds the committed model. To use another export:
   `python tools/select_model.py <export folder>` (copies `model.espdl`, writes `model/model_info.json`).
4. Build and flash (the port is e.g. `/dev/ttyACM0` on Linux, `/dev/cu.usbmodem*` on macOS, `COM3` on Windows):
   ```bash
   idf.py set-target esp32s3          # first time only
   idf.py -p <port> flash monitor     # everything, including the 2.9 MB model partition
   idf.py -p <port> app-flash monitor # code and web page only, when the model did not change
   ```
   `Ctrl+]` leaves the monitor. The log ends with the board's address (`ready: http://rabbitfish.local/`).

At boot the device:
1. joins Wi-Fi and starts the server first (they need internal RAM, the model can live in PSRAM);
2. loads the model from its flash partition and runs the **self-test** embedded in the `.espdl`
   (one val image and the outputs esp-ppq's simulator computed for it, compared bit for bit);
3. times one inference on one and on both cores and keeps the faster.

`POST /detect` answers "model still loading" for the few seconds before step 2 ends.

## Check accuracy on the device

```bash
python firmware/rabbitfish_server/tools/evaluate_device.py --split test     # needs the dataset
python firmware/rabbitfish_server/tools/evaluate_device.py --root data --split samples   # the 12 bundled images
```
Sends every image through `/detect` exactly as the page does and computes mAP the ultralytics way;
compare with the export's simulated numbers (`report.json`). Add `--host <ip>` if mDNS names do
not resolve on your network.

## Layout

- `main/detector.*`: JPEG decode, letterbox and quantise (int8 or int16 input), inference, box decoding and NMS
  (ports of `rabbitfish/data.py:letterbox` and `rabbitfish/metrics.py:decode`)
- `main/web_server.*`: the HTTP API and page; `main/index.html`: the page
- `main/wifi.*`: Wi-Fi station and mDNS (`rabbitfish.local`); on a failed join it scans and logs which networks it can see
- `partitions.csv`: 3 MB app, 6 MB `model` partition
- `sdkconfig.defaults`: 240 MHz, octal PSRAM at 80 MHz, Wi-Fi/lwIP buffers in PSRAM
- `tools/select_model.py`, `tools/evaluate_device.py`

Uses [ESP-DL](https://github.com/espressif/esp-dl) 3.3 (pulled by the component manager, `main/idf_component.yml`).
