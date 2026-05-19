"""
精确定位并裁剪纵断面图区域
"""
import cv2
import numpy as np
import os

INPUT_IMAGE = r"c:\yw\Project\jt-master\GL5.102.AZ-507(1)_images\page_0019.png"
OUTPUT_DIR = r"c:\yw\Project\jt-master\projects\elevation_detection\results\profile_cropped"

# 训练集尺寸
TARGET_WIDTH = 2382
TARGET_HEIGHT = 1685


def find_profile_graph(img):
    """定位纵断面图区域"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 检测水平线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # 检测竖直线
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # 合并水平线和竖直线
    combined = cv2.add(horizontal_lines, vertical_lines)

    # 膨胀连接断开的线
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=3)

    # 找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选大的矩形区域（纵断面图特征）
    profile_regions = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        # 纵断面图特征：面积大、宽高比合理
        if area > 100000 and cw > 2000 and ch > 500:
            aspect_ratio = cw / ch
            if 2 < aspect_ratio < 20:  # 宽图
                profile_regions.append({
                    'x': x, 'y': y, 'w': cw, 'h': ch,
                    'area': area, 'aspect': aspect_ratio
                })

    # 按面积排序
    profile_regions.sort(key=lambda r: r['area'], reverse=True)

    return profile_regions


def crop_profile_graph():
    """裁剪纵断面图"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取大图
    print("读取大图...")
    img = cv2.imread(INPUT_IMAGE)
    h, w = img.shape[:2]
    print(f"原始尺寸: {w} x {h}")

    # 定位纵断面图
    print("\n定位纵断面图区域...")
    regions = find_profile_graph(img)

    if not regions:
        print("未找到纵断面图区域，使用备选方案...")
        # 备选：检测包含大量水平线的区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # 检测水平线密度
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # 计算每行的水平线密度
        row_density = np.sum(horizontal_lines > 0, axis=1)

        # 找到密度最高的区域
        window_size = TARGET_HEIGHT
        max_density = 0
        best_y = 0
        for y in range(0, h - window_size, 100):
            density = np.sum(row_density[y:y+window_size])
            if density > max_density:
                max_density = density
                best_y = y

        regions = [{'x': 0, 'y': best_y, 'w': w, 'h': window_size, 'area': w * window_size, 'aspect': w / window_size}]

    print(f"找到 {len(regions)} 个候选区域")

    # 处理每个区域
    all_segments = []
    for i, region in enumerate(regions[:3]):  # 只处理前3个最大的
        print(f"\n区域 {i+1}: ({region['x']},{region['y']})-({region['x']+region['w']},{region['y']+region['h']})")
        print(f"  尺寸: {region['w']} x {region['h']}, 宽高比: {region['aspect']:.2f}")

        # 裁剪区域
        x1 = max(0, region['x'])
        y1 = max(0, region['y'])
        x2 = min(w, region['x'] + region['w'])
        y2 = min(h, region['y'] + region['h'])

        cropped = img[y1:y2, x1:x2]
        crop_path = os.path.join(OUTPUT_DIR, f"region_{i}_cropped.png")
        cv2.imwrite(crop_path, cropped)
        print(f"  已保存: {crop_path}")

        # 分段裁剪
        ch, cw = cropped.shape[:2]
        segments_x = max(1, cw // TARGET_WIDTH)
        segments_y = max(1, ch // TARGET_HEIGHT)

        seg_w = cw // segments_x if segments_x > 1 else cw
        seg_h = ch // segments_y if segments_y > 1 else ch

        print(f"  分段: {segments_y} x {segments_x}")

        for sy in range(segments_y):
            for sx in range(segments_x):
                y_start = sy * seg_h
                x_start = sx * seg_w
                y_end = min(y_start + seg_h, ch)
                x_end = min(x_start + seg_w, cw)

                segment = cropped[y_start:y_end, x_start:x_end]

                # 调整到训练集尺寸
                seg_h_actual, seg_w_actual = segment.shape[:2]
                if seg_w_actual != TARGET_WIDTH or seg_h_actual != TARGET_HEIGHT:
                    padded = np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8)
                    paste_x = max(0, (TARGET_WIDTH - seg_w_actual) // 2)
                    paste_y = max(0, (TARGET_HEIGHT - seg_h_actual) // 2)
                    paste_w = min(seg_w_actual, TARGET_WIDTH - paste_x)
                    paste_h = min(seg_h_actual, TARGET_HEIGHT - paste_y)
                    src_w = min(paste_w, segment.shape[1])
                    src_h = min(paste_h, segment.shape[0])
                    padded[paste_y:paste_y+src_h, paste_x:paste_x+src_w] = segment[:src_h, :src_w]
                    segment = padded

                seg_path = os.path.join(OUTPUT_DIR, f"region_{i}_seg_{sy}_{sx}.png")
                cv2.imwrite(seg_path, segment)
                all_segments.append(seg_path)

    print(f"\n完成！共生成 {len(all_segments)} 个分段")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    crop_profile_graph()
