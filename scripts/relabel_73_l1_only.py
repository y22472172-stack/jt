import argparse
import json
import shutil
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace only L1 VOC boxes using a YOLO model.")
    parser.add_argument("--model", type=Path, default=Path(r"C:\yw\Project\jt-master\best.pt"))
    parser.add_argument("--images-dir", type=Path, default=Path(r"C:\yw\Project\jt-master\73\images"))
    parser.add_argument("--xmls-dir", type=Path, default=Path(r"C:\yw\Project\jt-master\73\xmls"))
    parser.add_argument("--backup-root", type=Path, default=Path(r"C:\yw\Project\jt-master\73"))
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--class-name", default="L1")
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_xml(xmls_dir: Path, image_path: Path) -> Path:
    return xmls_dir / f"{image_path.stem}.xml"


def make_l1_object(template: ET.Element | None, class_name: str, bbox: list[int]) -> ET.Element:
    if template is not None:
        obj = deepcopy(template)
        obj.find("name").text = class_name
    else:
        obj = ET.Element("object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "bndbox")

    bndbox = obj.find("bndbox")
    if bndbox is None:
        bndbox = ET.SubElement(obj, "bndbox")
    for tag in ("xmin", "ymin", "xmax", "ymax"):
        if bndbox.find(tag) is None:
            ET.SubElement(bndbox, tag)
    for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), bbox):
        bndbox.find(tag).text = str(value)
    return obj


def update_xml(xml_path: Path, image_path: Path, detections: list[dict], class_name: str) -> tuple[int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    old_l1 = [obj for obj in root.findall("object") if obj.findtext("name") == class_name]
    template = old_l1[0] if old_l1 else None
    for obj in old_l1:
        root.remove(obj)

    for det in detections:
        root.append(make_l1_object(template, class_name, det["bbox"]))

    size = root.find("size")
    image = cv2.imread(str(image_path))
    if size is not None and image is not None:
        h, w = image.shape[:2]
        size.find("width").text = str(w)
        size.find("height").text = str(h)
        if size.find("depth") is not None:
            size.find("depth").text = str(image.shape[2] if len(image.shape) == 3 else 1)

    filename = root.find("filename")
    if filename is not None:
        filename.text = image_path.name
    path = root.find("path")
    if path is not None:
        path.text = str(image_path)

    ET.indent(tree, space="\t")
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)
    return len(old_l1), len(detections)


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_root / f"xmls_backup_before_l1_relabel_{timestamp}"
    shutil.copytree(args.xmls_dir, backup_dir)

    model = YOLO(str(args.model))
    target_class_ids = [idx for idx, name in model.names.items() if name == args.class_name]
    if not target_class_ids:
        raise RuntimeError(f"Class {args.class_name!r} not found in model names: {model.names}")
    target_class_id = target_class_ids[0]

    manifest = {
        "model": str(args.model),
        "images_dir": str(args.images_dir),
        "xmls_dir": str(args.xmls_dir),
        "backup_dir": str(backup_dir),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "class_name": args.class_name,
        "items": [],
    }

    total_old = 0
    total_new = 0
    for index, image_path in enumerate(list_images(args.images_dir), start=1):
        xml_path = find_xml(args.xmls_dir, image_path)
        if not xml_path.exists():
            print(f"[{index}] {image_path.name}: missing xml")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index}] {image_path.name}: unreadable image")
            continue
        h, w = image.shape[:2]

        detections = []
        for result in model.predict(source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                if class_id != target_class_id:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in box.xyxy[0].cpu().numpy().tolist()]
                x1 = max(0, min(w, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0].cpu().numpy()),
                    }
                )

        detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["confidence"]))
        old_count, new_count = update_xml(xml_path, image_path, detections, args.class_name)
        total_old += old_count
        total_new += new_count
        manifest["items"].append(
            {
                "image": str(image_path),
                "xml": str(xml_path),
                "old_l1_count": old_count,
                "new_l1_count": new_count,
                "detections": detections,
            }
        )
        print(f"[{index}/73] {image_path.name}: L1 {old_count} -> {new_count}")

    manifest_path = args.backup_root / f"l1_relabel_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Backup: {backup_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total L1: {total_old} -> {total_new}")


if __name__ == "__main__":
    main()
