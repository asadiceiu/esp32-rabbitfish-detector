"""Train one YOLO26 fish body/head detector (the step-by-step version is notebooks/01_train_yolo26.ipynb).

Recipe (the same one the YOLO11 baselines in results/baselines_yolo11 were trained with, so the
architectures can be compared directly):

    up to 500 epochs, early stop after 100 epochs without improvement (patience)
    AdamW, lr0 0.01, cosine schedule, seed 0, COCO-pretrained weights
    multi-scale training over 0.5x-1.5x of imgsz (ultralytics >= 8.4 writes this as multi_scale=0.5)

The best checkpoint is chosen on the val split; the test split is never touched here.

Output: runs/<name>/ (weights/best.pt, results.csv, plots) and runs/<name>/DONE. Re-running is
safe: a finished run is skipped and an interrupted one resumes from weights/last.pt.

Usage (from the repository root)
    python scripts/train_yolo26.py --size n --imgsz 320        # -> runs/yolo26n-b8-i320 (the deployed model)
    python scripts/train_yolo26.py --size n                     # -> runs/yolo26n-b8 (640 px)
    python scripts/train_yolo26.py --size n --epochs 2 --name smoke
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rabbitfish.paths import DATASET, RUNS, WEIGHTS  # noqa: E402  (sets YOLO_CONFIG_DIR)

from ultralytics import YOLO, settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", required=True, choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index, 'cpu' or 'mps'")
    parser.add_argument("--data", type=Path, default=DATASET / "data.yaml")
    parser.add_argument("--name", default=None,
                        help="run folder name; default yolo26<size>-b<batch>, plus -i<imgsz> when imgsz is not 640")
    args = parser.parse_args()

    name = args.name or f"yolo26{args.size}-b{args.batch}" + (f"-i{args.imgsz}" if args.imgsz != 640 else "")
    run_dir = RUNS / name
    done_marker = run_dir / "DONE"
    last = run_dir / "weights" / "last.pt"
    if done_marker.exists():
        print(f"{name}: already finished ({done_marker}), nothing to do")
        return
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found: put the dataset in dataset/ or set RABBITFISH_DATASET (see data/README.md)")

    # Pretrained weights live in weights/: download them once (e.g. on a cluster's login node)
    # and compute nodes need no internet. Only written when it differs.
    WEIGHTS.mkdir(exist_ok=True)
    if settings["weights_dir"] != str(WEIGHTS):
        settings.update({"weights_dir": str(WEIGHTS)})

    if last.exists():
        print(f"{name}: resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        pretrained = WEIGHTS / f"yolo26{args.size}.pt"
        print(f"{name}: training from {pretrained}")
        YOLO(str(pretrained) if pretrained.exists() else pretrained.name).train(  # a bare name downloads it
            data=str(args.data),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer="AdamW",
            lr0=0.01,
            cos_lr=True,
            multi_scale=0.5,
            seed=args.seed,
            workers=args.workers,
            device=args.device,
            project=str(RUNS),
            name=name,
            exist_ok=True,
        )

    done_marker.write_text("ok\n")
    print(f"{name}: finished, best checkpoint at {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
