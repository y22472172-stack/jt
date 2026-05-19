"""
XML标注转YOLO格式脚本
将VOC格式的XML标注转换为YOLO格式的txt标注，并划分训练集/验证集
"""
import os
import xml.etree.ElementTree as ET
import random
import shutil
from pathlib import Path

# 配置
RAW_IMAGES_DIR = r"c:\yw\Project\jt-master\236\images"
RAW_XMLS_DIR = r"c:\yw\Project\jt-master\236\xmls"
OUTPUT_DIR = r"c:\yw\Project\jt-master\projects\elevation_detection\data\yolo_dataset"
VAL_RATIO = 0.2  # 验证集比例
SEED = 42

# 类别映射 (根据你的标签调整)
CLASS_MAP = {
    'A': 0,  # 立面图
}
CLASS_NAMES = list(CLASS_MAP.keys())


def xml_to_yolo(xml_path, img_width, img_height):
    """将XML标注转换为YOLO格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP:
            continue

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换为YOLO格式 (center_x, center_y, width, height) 归一化到0-1
        cx = (xmin + xmax) / 2 / img_width
        cy = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        class_id = CLASS_MAP[name]
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def create_dataset():
    """创建YOLO数据集"""
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(RAW_XMLS_DIR) if f.endswith('.xml')]
    print(f"找到 {len(xml_files)} 个XML文件")

    # 随机划分
    random.seed(SEED)
    random.shuffle(xml_files)
    val_count = int(len(xml_files) * VAL_RATIO)
    val_files = set(xml_files[:val_count])
    train_files = set(xml_files[val_count:])

    print(f"训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")

    # 创建目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 转换标注
    converted = 0
    skipped = 0
    for xml_file in xml_files:
        xml_path = os.path.join(RAW_XMLS_DIR, xml_file)

        # 解析XML获取图片尺寸
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        if size is None:
            skipped += 1
            continue

        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 获取图片文件名
        filename = root.find('filename').text
        if not filename:
            filename = os.path.splitext(xml_file)[0] + '.png'

        # 检查图片是否存在
        img_path = os.path.join(RAW_IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在 {img_path}")
            skipped += 1
            continue

        # 确定是训练集还是验证集
        split = 'val' if xml_file in val_files else 'train'

        # 转换为YOLO格式
        yolo_lines = xml_to_yolo(xml_path, img_width, img_height)
        if not yolo_lines:
            skipped += 1
            continue

        # 复制图片
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, filename)
        shutil.copy2(img_path, dst_img_path)

        # 保存YOLO标注
        label_file = os.path.splitext(filename)[0] + '.txt'
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_file)
        with open(dst_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        converted += 1

    # 创建data.yaml
    yaml_content = f"""# YOLO Dataset Config
path: {OUTPUT_DIR}
train: images/train
val: images/val

# Classes
names:
  0: A  # 立面图
"""
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n转换完成:")
    print(f"- 成功转换: {converted} 张")
    print(f"- 跳过: {skipped} 张")
    print(f"- 数据集配置: {yaml_path}")


if __name__ == '__main__':
    create_dataset()
