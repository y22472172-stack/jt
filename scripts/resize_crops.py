import cv2

names = ['p19_top_left','p19_top_center','p19_top_right',
         'p19_left','p19_center_left','p19_center','p19_center_right','p19_right',
         'p19_table_left','p19_table_center','p19_table_right']

for name in names:
    img = cv2.imread(f'c:/Project/jt/{name}.png')
    h, w = img.shape[:2]
    scale = min(2000 / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imwrite(f'c:/Project/jt/{name}_sm.png', img)
    print(f'{name}: {w}x{h} -> {img.shape[1]}x{img.shape[0]}')
