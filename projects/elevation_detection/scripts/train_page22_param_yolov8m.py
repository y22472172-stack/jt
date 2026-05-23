"""
Retrain the page_0022 abutment parameter detector with YOLOv8m.

The dataset contains small dimension-text boxes on engineering drawings, so the
augmentation is intentionally conservative: no flips, light scale/translate,
and mosaic closed from the beginning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "sub_roi_dataset" / "lim" / "data_local.yaml"
DEFAULT_PROJECT = REPO_ROOT / "output" / "text_detection_retrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="lim_yolov8m")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        val=True,
        plots=True,
        seed=0,
        deterministic=True,
        cache=False,
        amp=True,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        translate=0.04,
        scale=0.25,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.08,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
