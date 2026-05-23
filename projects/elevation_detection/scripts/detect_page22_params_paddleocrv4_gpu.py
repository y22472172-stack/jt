"""
Detect page_0022 abutment parameters, then OCR them with PaddleOCRv4 GPU.

YOLO detection and PaddleOCR run in separate Python processes to avoid CUDA DLL
conflicts between torch and paddle on Windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "GL5.102.AZ-507(1)_images" / "page_0022.png"
DEFAULT_LIMIAN_MODEL = Path(r"c:\Users\ASUS\Desktop\output\limian\yolo\weights\best.pt")
DEFAULT_PARAM_MODEL = REPO_ROOT / "output" / "text_detection_73" / "yolov8m" / "weights" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0022_abutment_params_73_paddleocrv4_gpu"
OCR_HELPER = Path(__file__).with_name("paddleocrv4_gpu_manifest.py")

PARAM_ALIASES = {"D1": "柱宽", "L1": "柱距", "D2": "桩直径"}


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def expand_bbox(bbox: list[float], w: int, h: int, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return (
        clamp(x1 - pad_x, 0, w - 1),
        clamp(y1 - pad_y, 0, h - 1),
        clamp(x2 + pad_x, 1, w),
        clamp(y2 + pad_y, 1, h),
    )


def detect_best(model_path: Path, image: np.ndarray, imgsz: int, conf: float, iou: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detections = []
    for result in model.predict(source=image, imgsz=imgsz, conf=conf, iou=iou, verbose=False):
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            detections.append(
                {
                    "bbox": [float(v) for v in box.xyxy[0].cpu().numpy().tolist()],
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "class_id": class_id,
                    "class_name": str(model.names[class_id]),
                }
            )
    if not detections:
        raise RuntimeError(f"No detection from {model_path}")
    return max(detections, key=lambda item: item["confidence"])


def detect_params(model_path: Path, image: np.ndarray, imgsz: int, conf: float, iou: float) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detections = []
    for result in model.predict(source=image, imgsz=imgsz, conf=conf, iou=iou, verbose=False):
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = str(model.names[class_id])
            detections.append(
                {
                    "bbox": [float(v) for v in box.xyxy[0].cpu().numpy().tolist()],
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "class_id": class_id,
                    "class_name": class_name,
                    "alias": PARAM_ALIASES.get(class_name, class_name),
                }
            )
    detections.sort(key=lambda item: (item["class_id"], item["bbox"][1], item["bbox"][0]))
    return detections


def draw_box(img: np.ndarray, bbox: list[float], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--limian-model", type=Path, default=DEFAULT_LIMIAN_MODEL)
    parser.add_argument("--param-model", type=Path, default=DEFAULT_PARAM_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limian-conf", type=float, default=0.2)
    parser.add_argument("--param-conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--limian-imgsz", type=int, default=1280)
    parser.add_argument("--param-imgsz", type=int, default=1280)
    parser.add_argument("--limian-pad-x", type=int, default=20)
    parser.add_argument("--limian-pad-y", type=int, default=20)
    parser.add_argument("--ocr-pad-x", type=int, default=18)
    parser.add_argument("--ocr-pad-y", type=int, default=14)
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--ocr-scale", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    crop_dir = output_dir / "ocr_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    image_h, image_w = image.shape[:2]

    limian_det = detect_best(args.limian_model, image, args.limian_imgsz, args.limian_conf, args.iou)
    crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
        limian_det["bbox"], image_w, image_h, args.limian_pad_x, args.limian_pad_y
    )
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_path = output_dir / "page_0022_abutment_elevation_crop.png"
    cv2.imwrite(str(crop_path), crop)

    detections = detect_params(args.param_model, crop, args.param_imgsz, args.param_conf, args.iou)
    crop_h, crop_w = crop.shape[:2]
    manifest_items = []
    for index, det in enumerate(detections, start=1):
        ox1, oy1, ox2, oy2 = expand_bbox(det["bbox"], crop_w, crop_h, args.ocr_pad_x, args.ocr_pad_y)
        ocr_crop = crop[oy1:oy2, ox1:ox2].copy()
        ocr_crop_path = crop_dir / f"{index:02d}_{det['class_name']}_{det['confidence']:.2f}.png"
        cv2.imwrite(str(ocr_crop_path), ocr_crop)
        manifest_items.append(
            {
                **det,
                "bbox_abs": [
                    det["bbox"][0] + crop_x1,
                    det["bbox"][1] + crop_y1,
                    det["bbox"][2] + crop_x1,
                    det["bbox"][3] + crop_y1,
                ],
                "ocr_bbox_in_crop": [ox1, oy1, ox2, oy2],
                "ocr_crop": str(ocr_crop_path),
            }
        )

    manifest_path = output_dir / "ocr_manifest.json"
    manifest_path.write_text(json.dumps({"items": manifest_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    ocr_json = output_dir / "ocr_results_paddleocrv4_gpu.json"
    subprocess.run(
        [
            str(Path(sys.executable)),
            str(OCR_HELPER),
            "--manifest",
            str(manifest_path),
            "--output",
            str(ocr_json),
            "--device",
            args.ocr_device,
            "--ocr-scale",
            str(args.ocr_scale),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    ocr_items = json.loads(ocr_json.read_text(encoding="utf-8"))["items"]

    annotated_page = image.copy()
    annotated_crop = crop.copy()
    draw_box(annotated_page, [crop_x1, crop_y1, crop_x2, crop_y2], f"lim {limian_det['confidence']:.2f}", (0, 160, 0))
    colors = {"D1": (0, 120, 255), "D2": (255, 0, 160), "L1": (0, 180, 255)}
    for item in ocr_items:
        label = f"{item['class_name']} {item.get('numeric_value_text') or ''}".strip()
        color = colors.get(item["class_name"], (30, 30, 220))
        draw_box(annotated_crop, item["bbox"], label, color)
        draw_box(annotated_page, item["bbox_abs"], label, color)

    annotated_crop_path = output_dir / "page_0022_abutment_params_annotated_crop.png"
    annotated_page_path = output_dir / "page_0022_abutment_params_annotated_page.png"
    cv2.imwrite(str(annotated_crop_path), annotated_crop)
    cv2.imwrite(str(annotated_page_path), annotated_page)

    payload = {
        "image": str(args.image),
        "limian_model": str(args.limian_model),
        "param_model": str(args.param_model),
        "ocr_engine": "paddleocrv4",
        "ocr_device": args.ocr_device,
        "limian_detection": limian_det,
        "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
        "crop_path": str(crop_path),
        "detections": ocr_items,
    }
    json_path = output_dir / "page_0022_abutment_params.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "page_0022_abutment_params.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "class", "alias", "confidence", "numeric_value_text", "ocr_text", "bbox_abs"])
        for index, item in enumerate(ocr_items, start=1):
            writer.writerow(
                [
                    index,
                    item["class_name"],
                    item["alias"],
                    f"{item['confidence']:.6f}",
                    item.get("numeric_value_text") or "",
                    item.get("ocr_text") or "",
                    json.dumps(item["bbox_abs"], ensure_ascii=False),
                ]
            )

    print(f"Saved crop: {crop_path}")
    print(f"Saved annotated crop: {annotated_crop_path}")
    print(f"Saved annotated page: {annotated_page_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print("Detected parameters:")
    for item in ocr_items:
        print(
            f"  {item['class_name']}({item['alias']}) conf={item['confidence']:.3f} "
            f"value={item.get('numeric_value_text')} text={item.get('ocr_text')!r}"
        )


if __name__ == "__main__":
    main()
