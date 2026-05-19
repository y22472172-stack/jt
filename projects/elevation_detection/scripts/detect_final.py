"""
对裁剪后的纵断面图分段进行YOLO检测
"""
from ultralytics import YOLO
import cv2
import json
import os

INPUT_DIR = r"c:\yw\Project\jt-master\projects\elevation_detection\results\profile_cropped"
OUTPUT_DIR = r"c:\yw\Project\jt-master\projects\elevation_detection\results\detection_final"
MODEL_PATH = r"c:\yw\Project\jt-master\projects\elevation_detection\models\elevation_detect_v2\weights\best.pt"


def detect():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有分段
    segments = sorted([f for f in os.listdir(INPUT_DIR)
                      if f.startswith('region_') and f.endswith('.png') and 'seg' in f])
    print(f"共 {len(segments)} 个分段")

    all_detections = []

    for i, seg in enumerate(segments):
        # 每次重新加载模型避免状态问题
        model = YOLO(MODEL_PATH)

        seg_path = os.path.join(INPUT_DIR, seg)
        results = model.predict(seg_path, conf=0.15, verbose=False)

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                parts = seg.replace('.png', '').split('_')
                seg_idx = int(parts[1])
                seg_y = int(parts[3])
                seg_x = int(parts[4])

                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    all_detections.append({
                        'segment': seg,
                        'region': seg_idx,
                        'grid': [seg_x, seg_y],
                        'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                        'confidence': conf,
                        'class': model.names[cls_id]
                    })

                # 绘制检测结果
                img = cv2.imread(seg_path)
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                    label = f'A: {conf:.2f}'
                    cv2.putText(img, label, (bx1, by1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                out_path = os.path.join(OUTPUT_DIR, f'detected_{seg}')
                cv2.imwrite(out_path, img)

                print(f"  #{i+1} {seg}: {model.names[cls_id]} ({conf:.2f})")

        del model

    print(f"\n检测完成!")
    print(f"有检测的分段: {len(set(d['segment'] for d in all_detections))} 个")
    print(f"总检测数: {len(all_detections)} 个")

    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, "detections.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'detections': all_detections}, f, indent=2, ensure_ascii=False)
    print(f"检测数据: {json_path}")


if __name__ == '__main__':
    detect()
