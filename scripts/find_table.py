import cv2
import numpy as np

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]

# Convert to grayscale and find dark horizontal lines (table borders)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Look for horizontal lines by checking row-wise dark pixel density
# Table lines are typically long horizontal dark lines
for y_start in range(5000, 10000, 200):
    strip = gray[y_start:y_start+50, :]
    dark_ratio = np.mean(strip < 128)
    if dark_ratio > 0.15:
        print(f'y={y_start}: dark_ratio={dark_ratio:.3f}')

print("---")
# Also check vertical structure
for y_start in range(5000, 10000, 500):
    strip = gray[y_start:y_start+500, :]
    dark_ratio = np.mean(strip < 128)
    print(f'y={y_start}-{y_start+500}: dark_ratio={dark_ratio:.3f}')
