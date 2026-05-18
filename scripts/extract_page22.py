import fitz, numpy as np, cv2

doc = fitz.open('c:/Project/jt/GL5.102.AZ-507(1).pdf')
page = doc[21]

zoom = 600 / 72.0
matrix = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=matrix, alpha=False)
img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite('c:/Project/jt/page22_hires.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
print(f'Saved full: {pix.width}x{pix.height}')

h, w = img_bgr.shape[:2]
cv2.imwrite('c:/Project/jt/page22_crop_topleft.png', img_bgr[0:h//3, 0:w//2])
cv2.imwrite('c:/Project/jt/page22_crop_topright.png', img_bgr[0:h//3, w//2:w])
cv2.imwrite('c:/Project/jt/page22_crop_mid.png', img_bgr[h//4:h//2, :])
cv2.imwrite('c:/Project/jt/page22_crop_bottom.png', img_bgr[h//2:, :])
cv2.imwrite('c:/Project/jt/page22_crop_notes.png', img_bgr[h//3:2*h//3, w//2:w])

doc.close()
print('Done')
