import os
import json
import yaml
from PIL import Image
from tqdm import tqdm


def yolo_to_coco_for_detr():
    """
    将 YOLO 格式数据集转换为完全兼容 COCO/DETR 的标注格式
    """

    # 1️⃣ 读取 data.yaml 配置
    with open('yolo11n_data.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    # YOLO 中的类别名
    class_names = data_config.get('names', [])

    # ✅ 如果你希望手动替换成更有意义的类别名，可在这里设置
    meaningful_names = ['cc', 'cm', 'lt']

    assert len(class_names) == len(meaningful_names), \
        f"类别数量不匹配：data.yaml中有 {len(class_names)} 个类别，而 meaningful_names 有 {len(meaningful_names)} 个。"

    # 2️⃣ 输出目录
    output_dir = "coco_format_data"
    os.makedirs(output_dir, exist_ok=True)

    # 3️⃣ 数据集划分
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n🔄 正在处理 {split} 数据集...")

        # 构建 COCO 基础结构
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 添加类别信息（COCO 要求 id 从 1 开始）
        for i, (orig_name, meaningful_name) in enumerate(zip(class_names, meaningful_names)):
            coco_data["categories"].append({
                "id": i + 1,
                "name": meaningful_name,
                "supercategory": "object"
            })

        # 路径定义
        images_dir = f"yolo11n_data/{split}/images"
        labels_dir = f"yolo11n_data/{split}/labels"

        if not os.path.exists(images_dir):
            print(f"⚠️ 警告: {images_dir} 不存在，跳过此分割。")
            continue

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not image_files:
            print(f"⚠️ {images_dir} 中没有图像文件。")
            continue

        image_id = 0
        annotation_id = 0

        # 遍历图像文件
        for image_file in tqdm(image_files, desc=f"转换 {split}"):
            image_path = os.path.join(images_dir, image_file)

            # 读取图像尺寸
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"❌ 无法读取图像 {image_path}: {e}")
                continue

            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })

            # 对应标签路径
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            # 读取 YOLO 标签
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])

                    # YOLO -> COCO 坐标转换
                    x = (x_center - w / 2) * width
                    y = (y_center - h / 2) * height
                    bbox_width = w * width
                    bbox_height = h * height

                    # ✅ 边界约束（防止越界）
                    x = max(0, min(x, width - bbox_width))
                    y = max(0, min(y, height - bbox_height))
                    bbox_width = max(1.0, min(bbox_width, width - x))
                    bbox_height = max(1.0, min(bbox_height, height - y))

                    # 添加标注
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # ✅ COCO 类别ID从1开始
                        "bbox": [x, y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            image_id += 1

        # 保存 JSON 文件
        output_file = os.path.join(output_dir, f'{split}_annotations.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"✅ {split} 数据集转换完成：")
        print(f"  - 图像数量: {len(coco_data['images'])}")
        print(f"  - 标注数量: {len(coco_data['annotations'])}")
        print(f"  - 保存路径: {output_file}")

    print("\n🎯 所有数据集转换完成！COCO 格式完全兼容 DETR。")


if __name__ == "__main__":
    yolo_to_coco_for_detr()
