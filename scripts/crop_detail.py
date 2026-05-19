import cv2

img = cv2.imread('c:/Project/jt/GL5.102.AZ-507(1)_images/page_0019.png')
h, w = img.shape[:2]
print(f'Full: {w}x{h}')

# The bridge elevation is roughly in top 45% of image
# Full width is ~24578px

# 1. Span dimension chain at very top (full width, thin strip)
cv2.imwrite('c:/Project/jt/p19_span_dims.png', img[200:1200, 500:w-500])

# 2. Left abutment area (pier 0-1) with detailed annotations
cv2.imwrite('c:/Project/jt/p19_pier01_detail.png', img[1000:7000, 300:6500])

# 3. Pier 1-2 area (river section with water levels)
cv2.imwrite('c:/Project/jt/p19_pier12_detail.png', img[1000:7000, 5500:12000])

# 4. Pier 3-4 area
cv2.imwrite('c:/Project/jt/p19_pier34_detail.png', img[1000:7000, 11000:17000])

# 5. Pier 5-6-7 area (right side)
cv2.imwrite('c:/Project/jt/p19_pier567_detail.png', img[1000:7000, 16000:24000])

# 6. Full data table at bottom of drawing (before blank area)
cv2.imwrite('c:/Project/jt/p19_datatable.png', img[7000:10000, 300:w-300])

# 7. Water level / foundation area (middle vertical strip)
cv2.imwrite('c:/Project/jt/p19_foundation.png', img[4000:8500, 300:12000])

# 8. Right side table / notes
cv2.imwrite('c:/Project/jt/p19_notes_right.png', img[1000:5000, 20000:w-300])

print('Done - 8 detail crops saved')
