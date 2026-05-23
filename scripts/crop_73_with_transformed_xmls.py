import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop detected elevation drawings and transform VOC XML labels into crop coordinates."
    )
    parser.add_argument(
        "--model",
        default=r"c:\Users\ASUS\Desktop\output\limian\yolo\weights\best.pt",
        type=Path,
    )
    parser.add_argument(
        "--images-dir",
        default=r"c:\yw\Project\jt-master\73\images",
        type=Path,
    )
    parser.add_argument(
        "--xmls-dir",
        default=r"c:\yw\Project\jt-master\73\xmls",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=r"c:\yw\Project\jt-master\73\cropped_by_best",
        type=Path,
    )
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--pad", default=0, type=int, help="Optional crop padding in pixels.")
    parser.add_argument(
        "--fallback-full-image",
        action="store_true",
        help="When no detection exists, copy the full image and transform XML unchanged.",
    )
    return parser.parse_args()


def image_files(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def clamp_bbox(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def detections_for_image(
    model: YOLO,
    image,
    width: int,
    height: int,
    conf: float,
    iou: float,
    imgsz: int,
    pad: int,
) -> list[dict]:
    results = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            bbox = (
                int(round(x1)) - pad,
                int(round(y1)) - pad,
                int(round(x2)) + pad,
                int(round(y2)) + pad,
            )
            x1i, y1i, x2i, y2i = clamp_bbox(bbox, width, height)
            if x2i <= x1i or y2i <= y1i:
                continue
            detections.append(
                {
                    "bbox": [x1i, y1i, x2i, y2i],
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "class_id": int(box.cls[0].cpu().numpy()),
                    "class_name": result.names[int(box.cls[0].cpu().numpy())],
                }
            )
    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["confidence"]))
    return detections


def set_text(parent: ET.Element, path: str, value: str) -> None:
    node = parent.find(path)
    if node is not None:
        node.text = value


def transform_xml(
    xml_path: Path,
    source_image_path: Path,
    output_xml_path: Path,
    output_image_path: Path,
    crop_bbox: list[int],
    crop_width: int,
    crop_height: int,
) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    set_text(root, "folder", output_image_path.parent.name)
    set_text(root, "filename", output_image_path.name)
    set_text(root, "path", str(output_image_path))
    set_text(root, "size/width", str(crop_width))
    set_text(root, "size/height", str(crop_height))

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    kept_objects = 0
    for obj in list(root.findall("object")):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            root.remove(obj)
            continue

        xmin = int(round(float(bndbox.findtext("xmin", "0"))))
        ymin = int(round(float(bndbox.findtext("ymin", "0"))))
        xmax = int(round(float(bndbox.findtext("xmax", "0"))))
        ymax = int(round(float(bndbox.findtext("ymax", "0"))))

        ix1 = max(xmin, crop_x1)
        iy1 = max(ymin, crop_y1)
        ix2 = min(xmax, crop_x2)
        iy2 = min(ymax, crop_y2)
        if ix2 <= ix1 or iy2 <= iy1:
            root.remove(obj)
            continue

        bndbox.find("xmin").text = str(ix1 - crop_x1)
        bndbox.find("ymin").text = str(iy1 - crop_y1)
        bndbox.find("xmax").text = str(ix2 - crop_x1)
        bndbox.find("ymax").text = str(iy2 - crop_y1)
        kept_objects += 1

    comment = ET.Comment(
        f"source_image={source_image_path.name}; crop_bbox_xyxy={crop_x1},{crop_y1},{crop_x2},{crop_y2}"
    )
    root.insert(0, comment)

    ET.indent(tree, space="\t")
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    return kept_objects


def main() -> None:
    args = parse_args()
    crops_dir = args.output_dir / "images"
    out_xmls_dir = args.output_dir / "xmls"
    manifest_path = args.output_dir / "manifest.json"

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_xmls_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    manifest = []
    skipped = []
    total_crops = 0
    total_objects = 0

    files = image_files(args.images_dir)
    for index, image_path in enumerate(files, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append({"image": image_path.name, "reason": "could not read image"})
            print(f"[{index}/{len(files)}] {image_path.name}: skipped, could not read image")
            continue

        height, width = image.shape[:2]
        xml_path = args.xmls_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            skipped.append({"image": image_path.name, "reason": "missing xml"})
            print(f"[{index}/{len(files)}] {image_path.name}: skipped, missing XML")
            continue

        detections = detections_for_image(
            model=model,
            image=image,
            width=width,
            height=height,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            pad=args.pad,
        )

        if not detections and args.fallback_full_image:
            detections = [
                {
                    "bbox": [0, 0, width, height],
                    "confidence": 1.0,
                    "class_id": -1,
                    "class_name": "full_image_fallback",
                }
            ]

        if not detections:
            skipped.append({"image": image_path.name, "reason": "no detection"})
            print(f"[{index}/{len(files)}] {image_path.name}: no detection")
            continue

        image_crops = 0
        image_objects = 0
        for crop_index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            crop = image[y1:y2, x1:x2].copy()
            out_stem = f"{image_path.stem}_roi{crop_index}"
            out_image_path = crops_dir / f"{out_stem}.png"
            out_xml_path = out_xmls_dir / f"{out_stem}.xml"

            cv2.imwrite(str(out_image_path), crop)
            object_count = transform_xml(
                xml_path=xml_path,
                source_image_path=image_path,
                output_xml_path=out_xml_path,
                output_image_path=out_image_path,
                crop_bbox=detection["bbox"],
                crop_width=x2 - x1,
                crop_height=y2 - y1,
            )

            row = deepcopy(detection)
            row.update(
                {
                    "source_image": str(image_path),
                    "source_xml": str(xml_path),
                    "output_image": str(out_image_path),
                    "output_xml": str(out_xml_path),
                    "objects_kept": object_count,
                }
            )
            manifest.append(row)
            image_crops += 1
            image_objects += object_count

        total_crops += image_crops
        total_objects += image_objects
        print(
            f"[{index}/{len(files)}] {image_path.name}: crops={image_crops}, "
            f"objects={image_objects}"
        )

    manifest_path.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "images_dir": str(args.images_dir),
                "xmls_dir": str(args.xmls_dir),
                "output_dir": str(args.output_dir),
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "pad": args.pad,
                "crops": manifest,
                "skipped": skipped,
                "total_input_images": len(files),
                "total_output_crops": total_crops,
                "total_objects_kept": total_objects,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Done. input_images={len(files)}, output_crops={total_crops}, objects={total_objects}")
    print(f"Images: {crops_dir}")
    print(f"XMLs: {out_xmls_dir}")
    print(f"Manifest: {manifest_path}")
    if skipped:
        print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
