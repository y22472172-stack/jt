import cv2

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]

# The data table is in the lower portion of the drawing area
# Based on the full image, the table spans roughly y=7000-9500, full width
# Let me get the table with better vertical resolution

# Table left portion (row labels + first few piers)
table = img[6800:9200, 200:8000]
cv2.imwrite('c:/Project/jt/p19_table_crop1.png', table)

# Table center portion
table2 = img[6800:9200, 7500:16000]
cv2.imwrite('c:/Project/jt/p19_table_crop2.png', table2)

# Table right portion
table3 = img[6800:9200, 15500:24000]
cv2.imwrite('c:/Project/jt/p19_table_crop3.png', table3)

# Also get the span dimension chain more precisely
# Top of drawing has the span dimensions
dims = img[300:1800, 200:24000]
cv2.imwrite('c:/Project/jt/p19_dims_precise.png', dims)

# Get the pier elevation annotations (around y=2000-5000)
elev = img[1500:5500, 200:24000]
cv2.imwrite('c:/Project/jt/p19_elevations.png', elev)

print('Done')
