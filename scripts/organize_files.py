"""
整理项目目录：将临时裁切图片和辅助脚本分类归档。
"""
import os
import shutil

BASE = r'c:\Project\jt'

# 需要保留的主文件
KEEP_FILES = {'pdf_to_images.py'}

# 创建目录结构
dirs = {
    'temp_crops':      os.path.join(BASE, 'temp_crops'),       # 临时裁切图
    'temp_crops/p19':  os.path.join(BASE, 'temp_crops', 'p19'), # page19 裁切
    'temp_crops/p22':  os.path.join(BASE, 'temp_crops', 'p22'), # page22 裁切
    'scripts':         os.path.join(BASE, 'scripts'),           # 辅助脚本
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# 整理规则
moved = 0
for f in os.listdir(BASE):
    src = os.path.join(BASE, f)
    if not os.path.isfile(src):
        continue
    if f in KEEP_FILES:
        continue

    dst = None

    # page19 裁切图
    if f.startswith('p19_'):
        if f.endswith('_sm.png'):
            # 缩略图也放一起，加 _thumb 后缀方便区分
            dst = os.path.join(dirs['temp_crops/p19'], f)
        else:
            dst = os.path.join(dirs['temp_crops/p19'], f)

    # page22 裁切图
    elif f.startswith('page22_'):
        dst = os.path.join(dirs['temp_crops/p22'], f)

    # page19 预览图
    elif f.startswith('page19_'):
        dst = os.path.join(dirs['temp_crops/p19'], f)

    # 辅助脚本（非主脚本）
    elif f.endswith('.py') and f != 'pdf_to_images.py':
        dst = os.path.join(dirs['scripts'], f)

    if dst:
        shutil.move(src, dst)
        moved += 1
        print(f'  {f} -> {os.path.relpath(dst, BASE)}')

print(f'\n共整理 {moved} 个文件')

# 列出最终目录结构
print('\n=== 最终目录结构 ===')
for root, subdirs, files in os.walk(BASE):
    # 跳过增强后的图片目录和隐藏目录
    level = root.replace(BASE, '').count(os.sep)
    if level > 3:
        continue
    indent = '  ' * level
    folder_name = os.path.basename(root) or 'jt'
    print(f'{indent}{folder_name}/')
    subindent = '  ' * (level + 1)
    for file in sorted(files):
        fpath = os.path.join(root, file)
        size = os.path.getsize(fpath)
        if size > 1024*1024:
            size_str = f'{size/1024/1024:.1f}MB'
        elif size > 1024:
            size_str = f'{size/1024:.0f}KB'
        else:
            size_str = f'{size}B'
        print(f'{subindent}{file}  ({size_str})')
