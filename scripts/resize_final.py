import cv2

for name in ['p19_full_table', 'p19_table_left2', 'p19_table_mid2',
             'p19_table_right2', 'p19_table_far_right', 'p19_span_chain']:
    img = cv2.imread(f'c:/Project/jt/{name}.png')
    h, w = img.shape[:2]
    scale = min(2000 / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imwrite(f'c:/Project/jt/{name}_sm.png', img)
    print(f'{name}: {w}x{h} -> {img.shape[1]}x{img.shape[0]}')
