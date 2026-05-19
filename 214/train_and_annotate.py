"""
214数据集训练+自动标注全流程
1. VOC XML -> YOLO txt 转换 + 划分训练/验证集
2. 训练 YOLOv8m
3. 对未标注图片进行自动标注，输出VOC XML供人工修正
"""
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# ==================== 配置 ====================
BASE_DIR = r"c:\yw\Project\jt-master\214"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
XMLS_DIR = os.path.join(BASE_DIR, "xmls")
DATASET_DIR = os.path.join(BASE_DIR, "yolo_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "auto_labels")  # 自动标注XML输出
MODEL_NAME = "yolov8m.pt"
TRAIN_EPOCHS = 100
TRAIN_IMG_SIZE = 1280
TRAIN_BATCH = 4
CONF_THRESHOLD = 0.5
VAL_RATIO = 0.2
SEED = 42

# 类别映射（字母 -> 数字ID，保持字典序）
CLASS_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


def xml_to_yolo(xml_path, txt_path):
    """将单个VOC XML转换为YOLO txt"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # YOLO格式: class_id center_x center_y width height (归一化)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def yolo_to_xml(yolo_txt, img_path, xml_path, width, height, confidences=None):
    """将YOLO txt预测结果转回VOC XML"""
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = "images"
    fname = ET.SubElement(root, "filename")
    fname.text = os.path.basename(img_path)
    path_el = ET.SubElement(root, "path")
    path_el.text = img_path

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Auto-labeled by YOLOv8m"

    size_el = ET.SubElement(root, "size")
    for tag, val in [("width", width), ("height", height), ("depth", 3)]:
        el = ET.SubElement(size_el, tag)
        el.text = str(val)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    if not os.path.exists(yolo_txt):
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        return

    with open(yolo_txt, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else 1.0

        xmin = int((cx - bw / 2) * width)
        ymin = int((cy - bh / 2) * height)
        xmax = int((cx + bw / 2) * width)
        ymax = int((cy + bh / 2) * height)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        obj = ET.SubElement(root, "object")
        name_el = ET.SubElement(obj, "name")
        name_el.text = CLASS_NAMES.get(cls_id, str(cls_id))
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        for tag, val in [("xmin", xmin), ("ymin", ymin), ("xmax", xmax), ("ymax", ymax)]:
            el = ET.SubElement(bndbox, tag)
            el.text = str(val)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


# ==================== Step 1: 转换 + 划分 ====================
def step1_prepare_dataset():
    print("=" * 60)
    print("Step 1: VOC XML -> YOLO 格式转换 + 训练/验证集划分")
    print("=" * 60)

    # 获取所有已标注图片
    xml_files = [f for f in os.listdir(XMLS_DIR) if f.endswith(".xml")]
    print(f"共 {len(xml_files)} 个已标注XML")

    # 转换为YOLO格式
    all_data = []
    for xml_file in xml_files:
        stem = Path(xml_file).stem
        img_file = None
        for ext in [".png", ".jpg", ".jpeg"]:
            if os.path.exists(os.path.join(IMAGES_DIR, stem + ext)):
                img_file = stem + ext
                break
        if img_file is None:
            print(f"[警告] 找不到对应图片: {xml_file}")
            continue

        txt_path = os.path.join(DATASET_DIR, "labels_temp", f"{stem}.txt")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        xml_to_yolo(os.path.join(XMLS_DIR, xml_file), txt_path)
        all_data.append((img_file, txt_path))
        print(f"  转换: {xml_file} -> {stem}.txt")

    # 划分训练/验证集
    random.seed(SEED)
    random.shuffle(all_data)
    val_count = max(1, int(len(all_data) * VAL_RATIO))
    val_data = all_data[:val_count]
    train_data = all_data[val_count:]

    print(f"\n训练集: {len(train_data)} 张, 验证集: {val_count} 张")

    for split, data_list in [("train", train_data), ("val", val_data)]:
        img_dst = os.path.join(DATASET_DIR, "images", split)
        lbl_dst = os.path.join(DATASET_DIR, "labels", split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for img_file, txt_path in data_list:
            shutil.copy2(os.path.join(IMAGES_DIR, img_file), img_dst)
            shutil.copy2(txt_path, os.path.join(lbl_dst, Path(txt_path).name))

    # 创建data.yaml
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val
names:
  0: A
  1: B
  2: C
  3: D
"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml 已保存: {yaml_path}")

    # 清理临时目录
    temp_dir = os.path.join(DATASET_DIR, "labels_temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return yaml_path


# ==================== Step 2: 训练 ====================
def step2_train(yaml_path):
    print("\n" + "=" * 60)
    print("Step 2: 训练 YOLOv8m")
    print("=" * 60)

    model = YOLO(MODEL_NAME)
    results = model.train(
        data=yaml_path,
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMG_SIZE,
        batch=TRAIN_BATCH,
        name="detect_214",
        project=os.path.join(BASE_DIR, "models"),
        patience=30,
        degrees=5,
        shear=2,
        mixup=0.1,
        device=0,
    )

    best_weights = os.path.join(BASE_DIR, "models", "detect_214", "weights", "best.pt")
    print(f"训练完成! 最佳权重: {best_weights}")
    return best_weights


# ==================== Step 3: 自动标注 ====================
def step3_auto_label(weights_path):
    print("\n" + "=" * 60)
    print("Step 3: 自动标注未标注图片")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 已标注的图片名（不含扩展名）
    labeled_stems = {Path(f).stem for f in os.listdir(XMLS_DIR) if f.endswith(".xml")}
    print(f"已标注: {len(labeled_stems)} 张")

    # 获取未标注图片
    all_images = [f for f in os.listdir(IMAGES_DIR)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    unlabeled = [f for f in all_images if Path(f).stem not in labeled_stems]
    print(f"待自动标注: {len(unlabeled)} 张")

    model = YOLO(weights_path)

    for idx, img_file in enumerate(unlabeled):
        img_path = os.path.join(IMAGES_DIR, img_file)
        stem = Path(img_file).stem

        # 读取图片尺寸
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_file}")
            continue
        h, w = img.shape[:2]

        # YOLO检测，输出txt
        results = model.predict(
            img_path,
            conf=CONF_THRESHOLD,
            imgsz=TRAIN_IMG_SIZE,
            save_txt=True,
            save_conf=True,
            project=OUTPUT_DIR,
            name="predict_temp",
            exist_ok=True,
            verbose=False,
        )

        # 找到预测的txt文件
        pred_txt = os.path.join(OUTPUT_DIR, "predict_temp", "labels", f"{stem}.txt")

        # 转换为VOC XML
        xml_path = os.path.join(OUTPUT_DIR, f"{stem}.xml")
        yolo_to_xml(pred_txt, img_path, xml_path, w, h)

        # 统计检测结果
        if os.path.exists(pred_txt):
            with open(pred_txt) as f:
                count = len(f.readlines())
        else:
            count = 0

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(unlabeled)}] {img_file} -> {count} 个目标")

    # 清理临时预测目录
    temp_pred = os.path.join(OUTPUT_DIR, "predict_temp")
    if os.path.exists(temp_pred):
        shutil.rmtree(temp_pred)

    print(f"\n自动标注完成! XML保存在: {OUTPUT_DIR}")
    print(f"共标注 {len(unlabeled)} 张图片，请人工检查并修正后复制到 xmls/ 目录")


# ==================== 主流程 ====================
if __name__ == "__main__":
    yaml_path = step1_prepare_dataset()
    weights = step2_train(yaml_path)
    step3_auto_label(weights)
