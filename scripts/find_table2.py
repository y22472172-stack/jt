import cv2
import numpy as np

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Check overall histogram
print(f"Image shape: {w}x{h}")
print(f"Gray min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")

# Find where content actually is (non-white rows)
for y in range(0, h, 500):
    row_strip = gray[y:y+500, 1000:w-1000]
    non_white = np.mean(row_strip < 200)
    if non_white > 0.01:
        print(f"y={y}-{y+500}: content_ratio={non_white:.4f}")
