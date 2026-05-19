"""
Detect bridge pier numbers on profile crops produced by detect_page19_tiled.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "GL5.102.AZ-507(1)_images" / "page_0019.png"
DEFAULT_PROFILE_JSON = (
    REPO_ROOT
    / "projects"
    / "elevation_detection"
    / "results"
    / "page_0019_vertical_segments_opt"
    / "page_0019_tiled_detections.json"
)
DEFAULT_MODEL = REPO_ROOT / "projects" / "elevation_detection" / "models" / "pile.pt"
DEFAULT_OUTPUT = (
    REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_profile_piles"
)


def tile_positions(length: int, tile_length: int, overlap: int) -> list[int]:
    if length <= tile_length:
        return [0]
    step = tile_length - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile size")
    positions = list(range(0, max(1, length - tile_length + 1), step))
    last = length - tile_length
    if positions[-1] != last:
        positions.append(last)
    return positions


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    kept: list[dict] = []
    for class_id in sorted({det["class_id"] for det in detections}):
        pending = [det for det in detections if det["class_id"] == class_id]
        pending.sort(key=lambda det: det["confidence"], reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            best_box = np.array(best["bbox"], dtype=np.float32)
            pending = [
                det
                for det in pending
                if iou(best_box, np.array(det["bbox"], dtype=np.float32)) < iou_threshold
            ]
    kept.sort(key=lambda det: (det["profile_index"], det["bbox"][0], det["bbox"][1]))
    return kept


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )


def draw_box(img: np.ndarray, bbox: list[float], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y = max(0, y1 - th - baseline - 6)
    cv2.rectangle(img, (x1, label_y), (x1 + tw + 6, label_y + th + baseline + 6), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, label_y + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def detect_on_profile(
    model: YOLO,
    profile_img: np.ndarray,
    profile_index: int,
    profile_bbox: list[float],
    args: argparse.Namespace,
) -> list[dict]:
    height, width = profile_img.shape[:2]
    if args.resize_profile_width > 0 and args.resize_profile_height > 0:
        model_img = cv2.resize(
            profile_img,
            (args.resize_profile_width, args.resize_profile_height),
            interpolation=cv2.INTER_AREA,
        )
        results = model.predict(
            source=model_img,
            conf=args.conf,
            imgsz=args.imgsz,
            iou=args.model_iou,
            verbose=False,
        )

        scale_x = width / args.resize_profile_width
        scale_y = height / args.resize_profile_height
        detections: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                crop_bbox = [
                    float(bx1 * scale_x),
                    float(by1 * scale_y),
                    float(bx2 * scale_x),
                    float(by2 * scale_y),
                ]
                original_bbox = [
                    float(profile_bbox[0] + crop_bbox[0]),
                    float(profile_bbox[1] + crop_bbox[1]),
                    float(profile_bbox[0] + crop_bbox[2]),
                    float(profile_bbox[1] + crop_bbox[3]),
                ]
                detections.append(
                    {
                        "profile_index": profile_index,
                        "bbox": original_bbox,
                        "profile_bbox": crop_bbox,
                        "resized_input_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": str(model.names[class_id]),
                        "input_mode": "resized_profile",
                    }
                )
        return nms(detections, args.nms_iou)

    x_positions = tile_positions(width, args.tile_width, args.tile_overlap)
    y_positions = tile_positions(height, args.tile_height, args.tile_overlap)
    detections: list[dict] = []

    for tile_y in y_positions:
        for tile_x in x_positions:
            tile = profile_img[
                tile_y : tile_y + args.tile_height,
                tile_x : tile_x + args.tile_width,
            ]
            results = model.predict(
                source=tile,
                conf=args.conf,
                imgsz=args.imgsz,
                iou=args.model_iou,
                verbose=False,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    crop_bbox = [
                        float(tile_x + bx1),
                        float(tile_y + by1),
                        float(tile_x + bx2),
                        float(tile_y + by2),
                    ]
                    original_bbox = [
                        float(profile_bbox[0] + crop_bbox[0]),
                        float(profile_bbox[1] + crop_bbox[1]),
                        float(profile_bbox[0] + crop_bbox[2]),
                        float(profile_bbox[1] + crop_bbox[3]),
                    ]
                    detections.append(
                        {
                            "profile_index": profile_index,
                            "bbox": original_bbox,
                            "profile_bbox": crop_bbox,
                            "confidence": conf,
                            "class_id": class_id,
                            "class_name": str(model.names[class_id]),
                            "tile": [tile_x, tile_y],
                        }
                    )

    return nms(detections, args.nms_iou)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "profile_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    image_height, image_width = img.shape[:2]

    profile_data = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    profile_detections = profile_data["detections"]
    model = YOLO(str(args.model))
    print(f"Pile model classes: {model.names}")
    print(f"Profiles: {len(profile_detections)}")

    all_detections: list[dict] = []
    full_annotated = img.copy()

    for index, profile in enumerate(profile_detections, start=1):
        profile_bbox = profile.get("bbox_before_expand", profile["bbox"])
        x1, y1, x2, y2 = clamp_bbox(profile_bbox, image_width, image_height)
        profile_img = img[y1:y2, x1:x2].copy()

        crop_path = crops_dir / f"profile_{index:02d}.png"
        cv2.imwrite(str(crop_path), profile_img)
        if args.resize_profile_width > 0 and args.resize_profile_height > 0:
            resized_profile = cv2.resize(
                profile_img,
                (args.resize_profile_width, args.resize_profile_height),
                interpolation=cv2.INTER_AREA,
            )
            resized_path = crops_dir / f"profile_{index:02d}_resized_input.png"
            cv2.imwrite(str(resized_path), resized_profile)

        detections = detect_on_profile(
            model,
            profile_img,
            profile_index=index,
            profile_bbox=[x1, y1, x2, y2],
            args=args,
        )
        all_detections.extend(detections)

        crop_annotated = profile_img.copy()
        for det_index, det in enumerate(detections, start=1):
            draw_box(
                crop_annotated,
                det["profile_bbox"],
                f"{det_index} {det['confidence']:.2f}",
                (0, 180, 255),
            )
            draw_box(
                full_annotated,
                det["bbox"],
                f"P{index}-{det_index} {det['confidence']:.2f}",
                (0, 180, 255),
            )

        annotated_crop_path = crops_dir / f"profile_{index:02d}_piles.png"
        cv2.imwrite(str(annotated_crop_path), crop_annotated)
        print(f"Profile {index}: {len(detections)} pile detections")

    all_detections.sort(key=lambda det: (det["bbox"][0], det["bbox"][1]))

    annotated_path = output_dir / "page_0019_piles_detected.png"
    cv2.imwrite(str(annotated_path), full_annotated)

    json_path = output_dir / "page_0019_pile_detections.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(args.image),
                "profile_json": str(args.profile_json),
                "model": str(args.model),
                "conf": args.conf,
                "imgsz": args.imgsz,
                "tile_size": [args.tile_width, args.tile_height],
                "tile_overlap": args.tile_overlap,
                "resize_profile_to": (
                    [args.resize_profile_width, args.resize_profile_height]
                    if args.resize_profile_width > 0 and args.resize_profile_height > 0
                    else None
                ),
                "profile_count": len(profile_detections),
                "pile_detection_count": len(all_detections),
                "detections": all_detections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Total pile detections: {len(all_detections)}")
    print(f"Saved annotated image: {annotated_path}")
    print(f"Saved JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--model-iou", type=float, default=0.7)
    parser.add_argument("--nms-iou", type=float, default=0.4)
    parser.add_argument("--tile-width", type=int, default=2048)
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=300)
    parser.add_argument("--resize-profile-width", type=int, default=0)
    parser.add_argument("--resize-profile-height", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
