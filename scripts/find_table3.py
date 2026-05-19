import cv2
import numpy as np

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Content is in top ~5000px. Find table region more precisely
for y in range(0, 5500, 200):
    strip = gray[y:y+200, 1000:w-1000]
    non_white = np.mean(strip < 200)
    if non_white > 0.02:
        print(f"y={y}-{y+200}: content={non_white:.4f}")

# Also find horizontal lines (table rows) - look for rows with high dark pixel count
print("\n--- Horizontal line detection ---")
for y in range(0, 5500, 100):
    row = gray[y, 1000:w-1000]
    dark_pixels = np.sum(row < 100)
    total = len(row)
    ratio = dark_pixels / total
    if ratio > 0.3:
        print(f"y={y}: line_ratio={ratio:.3f}")
