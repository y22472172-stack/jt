import os
import xml.etree.ElementTree as ET

images_dir = r"c:\yw\Project\jt-master\236\images"
xmls_dir = r"c:\yw\Project\jt-master\236\xmls"

# 获取所有XML文件
xml_files = [f for f in os.listdir(xmls_dir) if f.endswith('.xml')]

deleted_count = 0
cleaned_count = 0
no_label_a_count = 0

for xml_file in xml_files:
    xml_path = os.path.join(xmls_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取所有object标签
    objects = root.findall('object')

    # 检查是否有标签A
    has_label_a = False
    for obj in objects:
        name = obj.find('name')
        if name is not None and name.text == 'A':
            has_label_a = True
            break

    if not has_label_a:
        # 没有标签A，删除图片和XML
        no_label_a_count += 1
        os.remove(xml_path)
        image_file = os.path.splitext(xml_file)[0] + '.png'
        image_path = os.path.join(images_dir, image_file)
        if os.path.exists(image_path):
            os.remove(image_path)
            deleted_count += 1
        print(f"删除无标签A: {xml_file}")
    else:
        # 有标签A，删除其他标签
        for obj in objects[:]:  # 使用切片创建副本以便修改
            name = obj.find('name')
            if name is not None and name.text != 'A':
                root.remove(obj)

        # 重新保存XML
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        cleaned_count += 1
        print(f"清理标签: {xml_file}")

print(f"\n处理完成:")
print(f"- 删除无标签A的图片: {deleted_count} 张")
print(f"- 清理XML中的其他标签: {cleaned_count} 个")
