import argparse
import json
import shutil
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer updated source-image L1 boxes to cropped_by_best XMLs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"C:\yw\Project\jt-master\73\cropped_by_best\manifest.json"),
    )
    parser.add_argument(
        "--source-xmls",
        type=Path,
        default=Path(r"C:\yw\Project\jt-master\73\xmls"),
    )
    parser.add_argument(
        "--cropped-xmls",
        type=Path,
        default=Path(r"C:\yw\Project\jt-master\73\cropped_by_best\xmls"),
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(r"C:\yw\Project\jt-master\73\cropped_by_best"),
    )
    parser.add_argument("--class-name", default="L1")
    return parser.parse_args()


def read_boxes(root: ET.Element, class_name: str) -> list[ET.Element]:
    return [obj for obj in root.findall("object") if obj.findtext("name") == class_name]


def bbox_from_object(obj: ET.Element) -> tuple[int, int, int, int]:
    box = obj.find("bndbox")
    if box is None:
        raise ValueError("object has no bndbox")
    return tuple(int(round(float(box.findtext(tag, "0")))) for tag in ("xmin", "ymin", "xmax", "ymax"))


def set_object_bbox(obj: ET.Element, bbox: tuple[int, int, int, int]) -> None:
    box = obj.find("bndbox")
    if box is None:
        box = ET.SubElement(obj, "bndbox")
    for tag in ("xmin", "ymin", "xmax", "ymax"):
        if box.find(tag) is None:
            ET.SubElement(box, tag)
    for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), bbox):
        box.find(tag).text = str(value)


def transform_source_object(
    source_obj: ET.Element,
    crop_bbox: list[int],
    crop_width: int,
    crop_height: int,
) -> ET.Element | None:
    sx1, sy1, sx2, sy2 = bbox_from_object(source_obj)
    cx1, cy1, cx2, cy2 = crop_bbox

    ix1 = max(sx1, cx1)
    iy1 = max(sy1, cy1)
    ix2 = min(sx2, cx2)
    iy2 = min(sy2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    new_bbox = (
        max(0, min(crop_width, ix1 - cx1)),
        max(0, min(crop_height, iy1 - cy1)),
        max(0, min(crop_width, ix2 - cx1)),
        max(0, min(crop_height, iy2 - cy1)),
    )
    if new_bbox[2] <= new_bbox[0] or new_bbox[3] <= new_bbox[1]:
        return None

    obj = deepcopy(source_obj)
    set_object_bbox(obj, new_bbox)
    return obj


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_root / f"xmls_backup_before_l1_transfer_{timestamp}"
    shutil.copytree(args.cropped_xmls, backup_dir)

    rows = []
    total_old = 0
    total_new = 0

    for item in manifest["crops"]:
        source_xml = args.source_xmls / Path(item["source_xml"]).name
        cropped_xml = args.cropped_xmls / Path(item["output_xml"]).name
        if not source_xml.exists() or not cropped_xml.exists():
            rows.append(
                {
                    "source_xml": str(source_xml),
                    "cropped_xml": str(cropped_xml),
                    "error": "missing xml",
                }
            )
            continue

        source_root = ET.parse(source_xml).getroot()
        cropped_tree = ET.parse(cropped_xml)
        cropped_root = cropped_tree.getroot()

        old_l1 = read_boxes(cropped_root, args.class_name)
        for obj in old_l1:
            cropped_root.remove(obj)

        crop_x1, crop_y1, crop_x2, crop_y2 = item["bbox"]
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        new_objects = []
        for source_obj in read_boxes(source_root, args.class_name):
            new_obj = transform_source_object(source_obj, item["bbox"], crop_width, crop_height)
            if new_obj is not None:
                new_objects.append(new_obj)
                cropped_root.append(new_obj)

        total_old += len(old_l1)
        total_new += len(new_objects)
        ET.indent(cropped_tree, space="\t")
        cropped_tree.write(cropped_xml, encoding="utf-8", xml_declaration=False)

        rows.append(
            {
                "source_xml": str(source_xml),
                "cropped_xml": str(cropped_xml),
                "crop_bbox": item["bbox"],
                "old_l1_count": len(old_l1),
                "new_l1_count": len(new_objects),
            }
        )
        print(f"{cropped_xml.name}: L1 {len(old_l1)} -> {len(new_objects)}")

    report_path = args.backup_root / f"l1_transfer_manifest_{timestamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "source_xmls": str(args.source_xmls),
                "cropped_xmls": str(args.cropped_xmls),
                "backup_dir": str(backup_dir),
                "class_name": args.class_name,
                "total_old_l1": total_old,
                "total_new_l1": total_new,
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Backup: {backup_dir}")
    print(f"Report: {report_path}")
    print(f"Total L1: {total_old} -> {total_new}")


if __name__ == "__main__":
    main()
