import cv2

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]

# Data table area: y=3600-5200, full width
table = img[3600:5200, 200:w-200]
cv2.imwrite('c:/Project/jt/p19_full_table.png', table)

# Left portion (row labels)
cv2.imwrite('c:/Project/jt/p19_table_left2.png', img[3600:5200, 200:6000])

# Middle portion
cv2.imwrite('c:/Project/jt/p19_table_mid2.png', img[3600:5200, 5500:14000])

# Right portion
cv2.imwrite('c:/Project/jt/p19_table_right2.png', img[3600:5200, 13500:22000])

# Far right (notes area)
cv2.imwrite('c:/Project/jt/p19_table_far_right.png', img[3600:5200, 21500:w-200])

# Also get the span dimension chain more precisely
cv2.imwrite('c:/Project/jt/p19_span_chain.png', img[600:1400, 200:w-200])

print('Done')
