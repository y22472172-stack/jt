import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert cropped_by_best VOC XMLs to a YOLO dataset.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"c:\yw\Project\jt-master\73\cropped_by_best"),
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=Path(r"c:\yw\Project\jt-master\73\predefined_classes.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"c:\yw\Project\jt-master\73\cropped_by_best_yolo_dataset"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_classes(classes_file: Path) -> list[str]:
    classes = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise RuntimeError(f"No classes found in {classes_file}")
    return classes


def convert_box(
    bndbox: ET.Element,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    xmin = float(bndbox.findtext("xmin", "0"))
    ymin = float(bndbox.findtext("ymin", "0"))
    xmax = float(bndbox.findtext("xmax", "0"))
    ymax = float(bndbox.findtext("ymax", "0"))

    xmin = max(0.0, min(float(image_width), xmin))
    xmax = max(0.0, min(float(image_width), xmax))
    ymin = max(0.0, min(float(image_height), ymin))
    ymax = max(0.0, min(float(image_height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None

    cx = ((xmin + xmax) / 2.0) / image_width
    cy = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return cx, cy, width, height


def image_path_for_xml(images_dir: Path, root: ET.Element, xml_path: Path) -> Path | None:
    filename = root.findtext("filename")
    candidates = []
    if filename:
        candidates.append(images_dir / filename)
    for suffix in IMAGE_SUFFIXES:
        candidates.append(images_dir / f"{xml_path.stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    source_images = args.source_dir / "images"
    source_xmls = args.source_dir / "xmls"
    output_dir = args.output_dir
    classes = read_classes(args.classes_file)
    class_map = {name: index for index, name in enumerate(classes)}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    xml_files = sorted(source_xmls.glob("*.xml"))
    random.Random(args.seed).shuffle(xml_files)
    val_count = max(1, int(round(len(xml_files) * args.val_ratio))) if len(xml_files) > 1 else 0
    val_set = set(xml_files[:val_count])

    converted = 0
    skipped = []
    object_counts = {name: 0 for name in classes}

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            skipped.append((xml_path.name, "missing size"))
            continue

        image_width = int(float(size.findtext("width", "0")))
        image_height = int(float(size.findtext("height", "0")))
        if image_width <= 0 or image_height <= 0:
            skipped.append((xml_path.name, "invalid size"))
            continue

        image_path = image_path_for_xml(source_images, root, xml_path)
        if image_path is None:
            skipped.append((xml_path.name, "missing image"))
            continue

        lines = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name", "").strip()
            if class_name not in class_map:
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            converted_box = convert_box(bndbox, image_width, image_height)
            if converted_box is None:
                continue
            cx, cy, width, height = converted_box
            lines.append(f"{class_map[class_name]} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            object_counts[class_name] += 1

        if not lines:
            skipped.append((xml_path.name, "no valid objects"))
            continue

        split = "val" if xml_path in val_set else "train"
        output_image = output_dir / "images" / split / image_path.name
        output_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        shutil.copy2(image_path, output_image)
        output_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        converted += 1

    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    data_yaml = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
{names_yaml}
"""
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"classes: {class_map}")
    print(f"converted: {converted}")
    print(f"train: {len(list((output_dir / 'images' / 'train').glob('*')))}")
    print(f"val: {len(list((output_dir / 'images' / 'val').glob('*')))}")
    print(f"objects: {object_counts}")
    print(f"data_yaml: {output_dir / 'data.yaml'}")
    if skipped:
        print(f"skipped: {len(skipped)}")
        for item in skipped[:20]:
            print(f"  {item[0]}: {item[1]}")


if __name__ == "__main__":
    main()
