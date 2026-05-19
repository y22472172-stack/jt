"""
Run the 214 detect_214 model on profile crops and keep only classes C and D.
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
DEFAULT_MODEL = REPO_ROOT / "214" / "models" / "detect_214" / "weights" / "best.pt"
DEFAULT_OUTPUT = (
    REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_detect_214_CD"
)


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(1, min(width, int(round(x2)))),
        max(1, min(height, int(round(y2)))),
    )


def draw_box(img, bbox: list[float], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    label_y = max(0, y1 - th - baseline - 6)
    cv2.rectangle(img, (x1, label_y), (x1 + tw + 8, label_y + th + baseline + 6), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 4, label_y + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def classwise_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
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
                if intersection_over_union(best_box, np.array(det["bbox"], dtype=np.float32))
                < iou_threshold
            ]
    kept.sort(key=lambda det: (det["bbox"][0], det["bbox"][1]))
    return kept


def is_edge_partial(det: dict, args: argparse.Namespace) -> bool:
    x1, _, x2, _ = det["profile_bbox"]
    crop_width, _ = det["profile_size"]
    width = x2 - x1
    touches_left = x1 <= args.edge_margin
    touches_right = crop_width - x2 <= args.edge_margin
    if not (touches_left or touches_right):
        return False

    return width < args.min_edge_box_width or det["confidence"] < args.edge_conf


def expand_detection(det: dict, image_width: int, image_height: int, ratio: float) -> dict:
    if ratio <= 0:
        return det

    expanded = dict(det)
    x1, y1, x2, y2 = det["bbox"]
    width = x2 - x1
    dx = width * ratio
    new_bbox = [
        float(max(0, x1 - dx)),
        float(max(0, y1)),
        float(min(image_width, x2 + dx)),
        float(min(image_height, y2)),
    ]

    origin_x, origin_y = det["profile_origin"]
    expanded["bbox_before_expand"] = det["bbox"]
    expanded["bbox"] = new_bbox
    expanded["profile_bbox_before_expand"] = det["profile_bbox"]
    expanded["profile_bbox"] = [
        float(new_bbox[0] - origin_x),
        float(new_bbox[1] - origin_y),
        float(new_bbox[2] - origin_x),
        float(new_bbox[3] - origin_y),
    ]
    return expanded


def passes_class_threshold(det: dict, class_conf_thresholds: dict[str, float]) -> bool:
    threshold = class_conf_thresholds.get(det["class_name"])
    return threshold is None or det["confidence"] >= threshold


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
    profiles = profile_data["detections"]

    model = YOLO(str(args.model))
    keep_names = set(args.keep_classes)
    keep_class_ids = [class_id for class_id, name in model.names.items() if name in keep_names]
    if not keep_class_ids:
        raise RuntimeError(f"No classes named {sorted(keep_names)} in model names: {model.names}")

    print(f"Model classes: {model.names}")
    print(f"Keeping classes: {keep_class_ids} ({', '.join(args.keep_classes)})")
    print(f"Profiles: {len(profiles)}")

    full_annotated = img.copy()
    raw_detections: list[dict] = []
    colors = {"C": (0, 180, 255), "D": (255, 80, 0)}
    profile_images: dict[int, np.ndarray] = {}

    for profile_index, profile in enumerate(profiles, start=1):
        profile_bbox = profile["bbox"] if args.use_expanded_profile_bbox else profile.get("bbox_before_expand", profile["bbox"])
        x1, y1, x2, y2 = clamp_bbox(profile_bbox, image_width, image_height)
        profile_img = img[y1:y2, x1:x2].copy()
        profile_images[profile_index] = profile_img
        crop_path = crops_dir / f"profile_{profile_index:02d}.png"
        cv2.imwrite(str(crop_path), profile_img)

        results = model.predict(
            source=profile_img,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=keep_class_ids,
            verbose=False,
        )

        profile_detections = []
        crop_height, crop_width = profile_img.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = str(model.names[class_id])
                confidence = float(box.conf[0].cpu().numpy())
                crop_bbox = [float(bx1), float(by1), float(bx2), float(by2)]
                original_bbox = [
                    float(x1 + bx1),
                    float(y1 + by1),
                    float(x1 + bx2),
                    float(y1 + by2),
                ]
                det = {
                    "profile_index": profile_index,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "profile_bbox": crop_bbox,
                    "bbox": original_bbox,
                    "profile_origin": [x1, y1],
                    "profile_size": [crop_width, crop_height],
                }
                profile_detections.append(det)
                raw_detections.append(det)

        print(f"Profile {profile_index}: {len(profile_detections)} raw C/D detections")

    class_conf_thresholds = {}
    for item in args.class_conf:
        class_name, value = item.split(":", 1)
        class_conf_thresholds[class_name] = float(value)

    class_filtered = [
        det for det in raw_detections if passes_class_threshold(det, class_conf_thresholds)
    ]
    edge_filtered = [det for det in class_filtered if not is_edge_partial(det, args)]
    nms_detections = classwise_nms(edge_filtered, args.global_nms_iou)
    all_detections = [
        expand_detection(det, image_width, image_height, args.expand_x_ratio)
        for det in nms_detections
    ]
    all_detections.sort(key=lambda det: (det["bbox"][0], det["bbox"][1]))

    detections_by_profile: dict[int, list[dict]] = {}
    for det in all_detections:
        detections_by_profile.setdefault(det["profile_index"], []).append(det)

    for profile_index, profile_img in profile_images.items():
        crop_annotated = profile_img.copy()
        profile_detections = detections_by_profile.get(profile_index, [])
        profile_detections.sort(key=lambda det: (det["profile_bbox"][0], det["profile_bbox"][1]))
        for det_index, det in enumerate(profile_detections, start=1):
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = colors.get(det["class_name"], (0, 180, 255))
            draw_box(crop_annotated, det["profile_bbox"], label, color)
            draw_box(full_annotated, det["bbox"], f"P{profile_index}-{det_index} {label}", color)

        annotated_crop_path = crops_dir / f"profile_{profile_index:02d}_CD.png"
        cv2.imwrite(str(annotated_crop_path), crop_annotated)
        print(f"Profile {profile_index}: {len(profile_detections)} final C/D detections")

    annotated_path = output_dir / "page_0019_detect_214_CD.png"
    cv2.imwrite(str(annotated_path), full_annotated)

    json_path = output_dir / "page_0019_detect_214_CD.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(args.image),
                "profile_json": str(args.profile_json),
                "model": str(args.model),
                "keep_classes": list(args.keep_classes),
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "profile_count": len(profiles),
                "raw_detection_count": len(raw_detections),
                "class_filtered_count": len(class_filtered),
                "edge_filtered_count": len(edge_filtered),
                "global_nms_count": len(nms_detections),
                "detection_count": len(all_detections),
                "edge_margin": args.edge_margin,
                "min_edge_box_width": args.min_edge_box_width,
                "edge_conf": args.edge_conf,
                "global_nms_iou": args.global_nms_iou,
                "expand_x_ratio": args.expand_x_ratio,
                "use_expanded_profile_bbox": args.use_expanded_profile_bbox,
                "class_conf_thresholds": class_conf_thresholds,
                "detections": all_detections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Raw C/D detections: {len(raw_detections)}")
    print(f"After class confidence filtering: {len(class_filtered)}")
    print(f"After edge filtering: {len(edge_filtered)}")
    print(f"After global NMS: {len(nms_detections)}")
    print(f"Total final C/D detections: {len(all_detections)}")
    print(f"Saved annotated image: {annotated_path}")
    print(f"Saved JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-classes", nargs="+", default=["C", "D"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--edge-margin", type=float, default=80)
    parser.add_argument("--min-edge-box-width", type=float, default=220)
    parser.add_argument("--edge-conf", type=float, default=0.7)
    parser.add_argument("--global-nms-iou", type=float, default=0.5)
    parser.add_argument("--expand-x-ratio", type=float, default=0.08)
    parser.add_argument("--use-expanded-profile-bbox", action="store_true", default=True)
    parser.add_argument("--use-tight-profile-bbox", dest="use_expanded_profile_bbox", action="store_false")
    parser.add_argument("--class-conf", nargs="*", default=["D:0.3"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
