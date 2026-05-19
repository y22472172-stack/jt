"""
YOLOv8训练脚本
训练立面图检测模型
"""
from ultralytics import YOLO
import os

# 配置
DATA_YAML = r"c:\yw\Project\jt-master\projects\elevation_detection\data\yolo_dataset\data.yaml"
OUTPUT_DIR = r"c:\yw\Project\jt-master\projects\elevation_detection\models"
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8


def train():
    """训练YOLOv8模型"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # nano版本，显存占用小

    # 开始训练
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='elevation_detect',
        project=OUTPUT_DIR,
        exist_ok=True,
        patience=20,  # 早停：20个epoch无提升则停止
        save=True,
        save_period=10,  # 每10个epoch保存一次
        verbose=True
    )

    print(f"\n训练完成!")
    print(f"模型保存在: {os.path.join(OUTPUT_DIR, 'elevation_detect')}")


if __name__ == '__main__':
    train()
