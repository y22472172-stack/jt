"""
Detect elevation/profile structures on a large page by cropping blank margins
and tiling the remaining content to the same size as the 236 training images.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = REPO_ROOT / "GL5.102.AZ-507(1)_images" / "page_0019.png"
DEFAULT_DATASET_IMAGES = REPO_ROOT / "236" / "images"
DEFAULT_MODEL = (
    REPO_ROOT
    / "projects"
    / "elevation_detection"
    / "models"
    / "elevation_detect_v2"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "projects" / "elevation_detection" / "results" / "page_0019_tiled"
)


def infer_training_size(images_dir: Path) -> tuple[int, int]:
    """Return the most common image size in the training image directory."""
    sizes: Counter[tuple[int, int]] = Counter()
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        sizes[(width, height)] += 1

    if not sizes:
        raise RuntimeError(f"No readable images found in {images_dir}")
    return sizes.most_common(1)[0][0]


def crop_non_white_content(
    img: np.ndarray,
    threshold: int = 245,
    margin: int = 30,
    min_component_area: int = 100,
    row_density_ratio: float = 0.0008,
    col_density_ratio: float = 0.0008,
    min_run: int = 20,
    tail_run_mass_ratio: float = 0.02,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop dense content and ignore sparse scan noise near page edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_height, img_width = img.shape[:2]

    row_counts = np.count_nonzero(mask, axis=1)
    col_counts = np.count_nonzero(mask, axis=0)
    row_threshold = max(10, int(img_width * row_density_ratio))
    col_threshold = max(10, int(img_height * col_density_ratio))

    def dense_runs(values: np.ndarray, threshold_value: int) -> list[tuple[int, int, int]]:
        dense = values >= threshold_value
        indices = np.flatnonzero(dense)
        if indices.size == 0:
            return []

        runs: list[tuple[int, int, int]] = []
        start = int(indices[0])
        prev = int(indices[0])
        for value in indices[1:]:
            current = int(value)
            if current == prev + 1:
                prev = current
                continue
            if prev - start + 1 >= min_run:
                runs.append((start, prev, int(values[start : prev + 1].sum())))
            start = current
            prev = current
        if prev - start + 1 >= min_run:
            runs.append((start, prev, int(values[start : prev + 1].sum())))

        if not runs:
            return [(int(indices[0]), int(indices[-1]), int(values[indices].sum()))]

        return runs

    def all_run_bounds(runs: list[tuple[int, int, int]]) -> tuple[int, int] | None:
        if not runs:
            return None
        return min(run[0] for run in runs), max(run[1] for run in runs)

    def main_content_row_bounds(runs: list[tuple[int, int, int]]) -> tuple[int, int] | None:
        if not runs:
            return None

        main_index = max(range(len(runs)), key=lambda index: runs[index][2])
        main_mass = runs[main_index][2]
        significant_tail_mass = main_mass * tail_run_mass_ratio

        kept = []
        for index, run in enumerate(runs):
            if index <= main_index or run[2] >= significant_tail_mass:
                kept.append(run)

        return min(run[0] for run in kept), max(run[1] for run in kept)

    row_bounds = main_content_row_bounds(dense_runs(row_counts, row_threshold))
    col_bounds = all_run_bounds(dense_runs(col_counts, col_threshold))

    if row_bounds is not None and col_bounds is not None:
        y1 = max(0, row_bounds[0] - margin)
        y2 = min(img_height, row_bounds[1] + margin + 1)
        x1 = max(0, col_bounds[0] - margin)
        x2 = min(img_width, col_bounds[1] + margin + 1)
        return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    # Fallback for unusual inputs: use connected non-white content.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_component_area:
            points.append(contour)

    if not points:
        return img.copy(), (0, 0, img_width, img_height)

    all_points = np.vstack(points)
    x, y, width, height = cv2.boundingRect(all_points)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_width, x + width + margin)
    y2 = min(img_height, y + height + margin)

    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def tile_positions(length: int, tile_length: int, overlap: int) -> list[int]:
    """Generate positions that cover the full length and keep every tile full size."""
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


def vertical_segment_width(crop_height: int, target_width: int, target_height: int) -> int:
    """Use a source segment width with the same aspect ratio as the training size."""
    return max(1, int(round(crop_height * target_width / target_height)))


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def classwise_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    kept: list[dict] = []
    for class_id in sorted({det["class_id"] for det in detections}):
        same_class = [det for det in detections if det["class_id"] == class_id]
        same_class.sort(key=lambda det: det["confidence"], reverse=True)

        while same_class:
            best = same_class.pop(0)
            kept.append(best)
            best_box = np.array(best["bbox"], dtype=np.float32)
            same_class = [
                det
                for det in same_class
                if intersection_over_union(best_box, np.array(det["bbox"], dtype=np.float32))
                < iou_threshold
            ]

    kept.sort(key=lambda det: (det["bbox"][1], det["bbox"][0], -det["confidence"]))
    return kept


def filter_by_box_shape(
    detections: list[dict],
    min_aspect: float,
    min_width: float,
    min_height: float,
) -> list[dict]:
    """Keep detections whose shape is plausible for long profile regions."""
    if min_aspect <= 0 and min_width <= 0 and min_height <= 0:
        return detections

    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        aspect = width / height if height > 0 else 0.0
        det["box_width"] = float(width)
        det["box_height"] = float(height)
        det["box_aspect"] = float(aspect)

        if min_aspect > 0 and aspect < min_aspect:
            continue
        if min_width > 0 and width < min_width:
            continue
        if min_height > 0 and height < min_height:
            continue
        filtered.append(det)
    return filtered


def expand_detections(
    detections: list[dict],
    image_width: int,
    image_height: int,
    expand_x_ratio: float,
    expand_y_ratio: float,
) -> list[dict]:
    """Expand boxes after detection to cover clipped edges between segments."""
    if expand_x_ratio <= 0 and expand_y_ratio <= 0:
        return detections

    expanded = []
    for det in detections:
        new_det = dict(det)
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        dx = width * expand_x_ratio
        dy = height * expand_y_ratio
        new_det["bbox_before_expand"] = det["bbox"]
        new_det["bbox"] = [
            float(max(0, x1 - dx)),
            float(max(0, y1 - dy)),
            float(min(image_width, x2 + dx)),
            float(min(image_height, y2 + dy)),
        ]
        expanded.append(new_det)
    return expanded


def draw_detections(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    result = img.copy()
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 180, 0), 3)

        label = f"{index} {det['class_name']} {det['confidence']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        label_y1 = max(0, y1 - text_h - baseline - 8)
        cv2.rectangle(
            result,
            (x1, label_y1),
            (x1 + text_w + 8, label_y1 + text_h + baseline + 8),
            (0, 180, 0),
            -1,
        )
        cv2.putText(
            result,
            label,
            (x1 + 4, label_y1 + text_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result


def run_detection(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    tiles_dir = output_dir / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_width, tile_height = infer_training_size(Path(args.dataset_images))
    print(f"Training tile size: {tile_width} x {tile_height}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read input image: {args.image}")
    original_height, original_width = image.shape[:2]
    print(f"Input image size: {original_width} x {original_height}")

    cropped, crop_bbox = crop_non_white_content(
        image,
        threshold=args.white_threshold,
        margin=args.blank_margin,
        min_component_area=args.min_component_area,
        row_density_ratio=args.row_density_ratio,
        col_density_ratio=args.col_density_ratio,
        min_run=args.min_dense_run,
        tail_run_mass_ratio=args.tail_run_mass_ratio,
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    cropped_height, cropped_width = cropped.shape[:2]
    print(
        "Content crop: "
        f"({crop_x1}, {crop_y1}) - ({crop_x2}, {crop_y2}), "
        f"{cropped_width} x {cropped_height}"
    )

    cropped_path = output_dir / "content_crop.png"
    cv2.imwrite(str(cropped_path), cropped)

    if args.vertical_only:
        source_tile_width = min(
            cropped_width, vertical_segment_width(cropped_height, tile_width, tile_height)
        )
        source_tile_height = cropped_height
        x_positions = tile_positions(cropped_width, source_tile_width, args.overlap)
        y_positions = [0]
        print(
            "Vertical-only segments: "
            f"1 row x {len(x_positions)} cols = {len(x_positions)}"
        )
        print(
            "Source segment size before resize: "
            f"{source_tile_width} x {source_tile_height}; "
            f"model input size: {tile_width} x {tile_height}"
        )
    else:
        source_tile_width = tile_width
        source_tile_height = tile_height
        x_positions = tile_positions(cropped_width, source_tile_width, args.overlap)
        y_positions = tile_positions(cropped_height, source_tile_height, args.overlap)
        print(
            f"Tiles: {len(y_positions)} rows x {len(x_positions)} cols = "
            f"{len(y_positions) * len(x_positions)}"
        )

    model = YOLO(str(args.model))
    print(f"Model classes: {model.names}")

    raw_detections: list[dict] = []
    tile_records: list[dict] = []

    for row, tile_y in enumerate(y_positions):
        for col, tile_x in enumerate(x_positions):
            tile = cropped[
                tile_y : tile_y + source_tile_height,
                tile_x : tile_x + source_tile_width,
            ]
            source_height, source_width = tile.shape[:2]
            model_tile = (
                cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                if args.vertical_only
                else tile
            )
            tile_name = f"tile_r{row:02d}_c{col:02d}_x{tile_x}_y{tile_y}.png"
            tile_path = tiles_dir / tile_name
            cv2.imwrite(str(tile_path), model_tile)

            tile_records.append(
                {
                    "row": row,
                    "col": col,
                    "tile_path": str(tile_path),
                    "source_size": [source_width, source_height],
                    "model_input_size": [tile_width, tile_height],
                    "crop_bbox": [
                        tile_x,
                        tile_y,
                        tile_x + source_width,
                        tile_y + source_height,
                    ],
                    "original_bbox": [
                        crop_x1 + tile_x,
                        crop_y1 + tile_y,
                        crop_x1 + tile_x + source_width,
                        crop_y1 + tile_y + source_height,
                    ],
                }
            )

            results = model.predict(
                source=model_tile,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.model_iou,
                verbose=False,
            )

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                    scale_x = source_width / tile_width
                    scale_y = source_height / tile_height
                    sx1 = bx1 * scale_x
                    sy1 = by1 * scale_y
                    sx2 = bx2 * scale_x
                    sy2 = by2 * scale_y
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    raw_detections.append(
                        {
                            "bbox": [
                                float(crop_x1 + tile_x + sx1),
                                float(crop_y1 + tile_y + sy1),
                                float(crop_x1 + tile_x + sx2),
                                float(crop_y1 + tile_y + sy2),
                            ],
                            "crop_bbox": [
                                float(tile_x + sx1),
                                float(tile_y + sy1),
                                float(tile_x + sx2),
                                float(tile_y + sy2),
                            ],
                            "model_input_bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": str(model.names[class_id]),
                            "tile": {"row": row, "col": col, "x": tile_x, "y": tile_y},
                        }
                    )

        print(f"Finished row {row + 1}/{len(y_positions)}")

    shape_filtered_detections = filter_by_box_shape(
        raw_detections,
        min_aspect=args.min_box_aspect,
        min_width=args.min_box_width,
        min_height=args.min_box_height,
    )
    detections = classwise_nms(shape_filtered_detections, args.nms_iou)
    detections = expand_detections(
        detections,
        image_width=original_width,
        image_height=original_height,
        expand_x_ratio=args.expand_x_ratio,
        expand_y_ratio=args.expand_y_ratio,
    )
    print(f"Raw detections: {len(raw_detections)}")
    print(f"Detections after shape filter: {len(shape_filtered_detections)}")
    print(f"Final detections after NMS: {len(detections)}")

    annotated = draw_detections(image, detections)
    annotated_path = output_dir / "page_0019_tiled_detected.png"
    cv2.imwrite(str(annotated_path), annotated)

    result = {
        "image": str(args.image),
        "model": str(args.model),
        "dataset_images": str(args.dataset_images),
        "original_size": [original_width, original_height],
        "content_crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
        "content_crop_size": [cropped_width, cropped_height],
        "vertical_only": args.vertical_only,
        "source_tile_size": [source_tile_width, source_tile_height],
        "model_input_size": [tile_width, tile_height],
        "overlap": args.overlap,
        "confidence_threshold": args.conf,
        "model_iou": args.model_iou,
        "nms_iou": args.nms_iou,
        "tiles": tile_records,
        "raw_detection_count": len(raw_detections),
        "shape_filtered_detection_count": len(shape_filtered_detections),
        "final_detection_count": len(detections),
        "min_box_aspect": args.min_box_aspect,
        "min_box_width": args.min_box_width,
        "min_box_height": args.min_box_height,
        "expand_x_ratio": args.expand_x_ratio,
        "expand_y_ratio": args.expand_y_ratio,
        "detections": detections,
    }

    json_path = output_dir / "page_0019_tiled_detections.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved crop: {cropped_path}")
    print(f"Saved annotated image: {annotated_path}")
    print(f"Saved JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--dataset-images", type=Path, default=DEFAULT_DATASET_IMAGES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--model-iou", type=float, default=0.7)
    parser.add_argument("--nms-iou", type=float, default=0.4)
    parser.add_argument("--overlap", type=int, default=1200)
    parser.add_argument("--vertical-only", action="store_true", default=True)
    parser.add_argument("--grid-tiles", dest="vertical_only", action="store_false")
    parser.add_argument("--min-box-aspect", type=float, default=1.8)
    parser.add_argument("--min-box-width", type=float, default=0)
    parser.add_argument("--min-box-height", type=float, default=0)
    parser.add_argument("--expand-x-ratio", type=float, default=0.06)
    parser.add_argument("--expand-y-ratio", type=float, default=0.0)
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--blank-margin", type=int, default=30)
    parser.add_argument("--min-component-area", type=int, default=100)
    parser.add_argument("--row-density-ratio", type=float, default=0.0008)
    parser.add_argument("--col-density-ratio", type=float, default=0.0008)
    parser.add_argument("--min-dense-run", type=int, default=20)
    parser.add_argument("--tail-run-mass-ratio", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run_detection(parse_args())
