"""
使用训练好的YOLOv8模型检测236/images中的图片，
将检测到的目标裁剪下来保存到lmt文件夹
"""
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# 配置
MODEL_PATH = r"c:\yw\Project\jt-master\projects\elevation_detection\models\elevation_detect_v2\weights\best.pt"
IMAGES_DIR = r"c:\yw\Project\jt-master\236\images"
OUTPUT_DIR = r"c:\yw\Project\jt-master\236\lmt"
CONF_THRESHOLD = 0.5


def detect_and_crop():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print("加载模型...")
    model = YOLO(MODEL_PATH)

    # 获取所有图片
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    print(f"共找到 {len(image_files)} 张图片")

    total_crops = 0

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_name}")
            continue

        # 检测
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)

        crops = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    # 边界检查
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img.shape[1], x2)
                    y2 = min(img.shape[0], y2)
                    crops.append((x1, y1, x2, y2, conf))

        if not crops:
            print(f"[{idx+1}/{len(image_files)}] {img_name} -> 未检测到目标")
            continue

        # 裁剪并保存
        stem = Path(img_name).stem
        for ci, (x1, y1, x2, y2, conf) in enumerate(crops):
            crop_img = img[y1:y2, x1:x2]
            if len(crops) == 1:
                out_name = f"{stem}.png"
            else:
                out_name = f"{stem}_{ci}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, crop_img)
            total_crops += 1

        print(f"[{idx+1}/{len(image_files)}] {img_name} -> 裁剪 {len(crops)} 个目标")

    print(f"\n完成! 共裁剪 {total_crops} 张图片，保存在: {OUTPUT_DIR}")


if __name__ == '__main__':
    detect_and_crop()
