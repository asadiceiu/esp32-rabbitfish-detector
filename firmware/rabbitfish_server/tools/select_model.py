"""Copy one esp32_export.py output into the firmware: model/model.espdl + model/model_info.json.

model_info.json is embedded in the firmware: the server reads the class names, strides and NMS
setting from it and returns it from GET /info, so the device always says which model it runs.

Usage (from the repository root):
    python firmware/rabbitfish_server/tools/select_model.py models/yolo26n-i320-esp32s3            # the committed model
    python firmware/rabbitfish_server/tools/select_model.py runs/yolo26n-b8-i320/esp32/320x256-w8a8-o2m-c10-box16-br
then `idf.py flash` (the model partition is rewritten with the new file).
"""
import argparse
import json
import shutil
from pathlib import Path

FW = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path, help="runs/<run>/esp32/<variant>/ with model.espdl and report.json")
    args = parser.parse_args()

    report = json.loads((args.export_dir / "report.json").read_text())
    if report["head"] != "o2m":
        raise SystemExit("the firmware decoder applies NMS and expects --head o2m exports")
    classes = [report["classes"][str(i)] if str(i) in report["classes"] else report["classes"][i]
               for i in range(len(report["classes"]))]
    info = {
        "run": report["run"],
        "variant": report.get("variant", args.export_dir.name),
        "input": {"width": report["input"]["width"], "height": report["input"]["height"],
                  "letterbox_pad": report["input"]["letterbox_pad"]},
        "classes": classes,
        "strides": report["strides"],
        "nms_iou": 0.7,
        "val_mAP50-95": {"float": round(report["mAP_float"]["mAP50-95"], 4),
                         "int8_simulated": round(report["mAP_quant"]["mAP50-95"], 4)},
        "val_mAP50": {"float": round(report["mAP_float"]["mAP50"], 4),
                      "int8_simulated": round(report["mAP_quant"]["mAP50"], 4)},
        "self_test_image": report.get("test_image"),
    }
    out = FW / "model"
    out.mkdir(exist_ok=True)
    shutil.copyfile(args.export_dir / "model.espdl", out / "model.espdl")
    (out / "model_info.json").write_text(json.dumps(info, indent=1))
    print(f"{args.export_dir} -> {out} ({(out / 'model.espdl').stat().st_size} bytes)")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
