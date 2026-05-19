import cv2
import numpy as np

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]
print(f'Full size: {w}x{h}')

# The bridge drawing is in the top ~60% of the image
# Bottom is mostly blank
bridge_h = int(h * 0.55)

# Crop horizontal strips across the bridge elevation
# Left portion (abutment 0, first few spans)
cv2.imwrite('c:/Project/jt/p19_left.png', img[0:bridge_h, 0:w//4])
# Center-left
cv2.imwrite('c:/Project/jt/p19_center_left.png', img[0:bridge_h, w//6:w//3])
# Center
cv2.imwrite('c:/Project/jt/p19_center.png', img[0:bridge_h, w//3:w//2])
# Center-right
cv2.imwrite('c:/Project/jt/p19_center_right.png', img[0:bridge_h, w//2:2*w//3])
# Right portion (last spans, abutment)
cv2.imwrite('c:/Project/jt/p19_right.png', img[0:bridge_h, 3*w//4:w])

# Also crop the data table area (bottom of drawing, before blank area)
# The table with pier data is usually at the very bottom of the drawing
table_top = int(h * 0.35)
table_bot = int(h * 0.55)
cv2.imwrite('c:/Project/jt/p19_table_left.png', img[table_top:table_bot, 0:w//3])
cv2.imwrite('c:/Project/jt/p19_table_center.png', img[table_top:table_bot, w//3:2*w//3])
cv2.imwrite('c:/Project/jt/p19_table_right.png', img[table_top:table_bot, 2*w//3:w])

# Top strip - elevation annotations
cv2.imwrite('c:/Project/jt/p19_top_left.png', img[0:int(h*0.18), 0:w//3])
cv2.imwrite('c:/Project/jt/p19_top_center.png', img[0:int(h*0.18), w//3:2*w//3])
cv2.imwrite('c:/Project/jt/p19_top_right.png', img[0:int(h*0.18), 2*w//3:w])

print('Done - 10 crops saved')
