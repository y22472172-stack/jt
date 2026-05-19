import cv2

names = ['p19_span_dims', 'p19_pier01_detail', 'p19_pier12_detail',
         'p19_pier34_detail', 'p19_pier567_detail', 'p19_datatable',
         'p19_foundation', 'p19_notes_right']

for name in names:
    img = cv2.imread(f'c:/Project/jt/{name}.png')
    h, w = img.shape[:2]
    scale = min(2000 / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imwrite(f'c:/Project/jt/{name}_sm.png', img)
    print(f'{name}: {w}x{h} -> {img.shape[1]}x{img.shape[0]}')
