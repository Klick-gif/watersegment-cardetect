import os
import cv2
import numpy as np
import yaml
from pathlib import Path


class YOLOToSegNetConverter:
    """YOLO 分割格式 → SegNet mask 图像格式转换"""

    def __init__(self, data_yaml):
        with open(data_yaml, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)

        self.base_path = Path(self.data_config['path'])
        self.num_classes = self.data_config['nc']

    def convert_single_label(self, label_path, img_shape, output_path):
        """将单个 YOLO 标签转换为 mask 图"""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if not os.path.exists(label_path):
            cv2.imwrite(str(output_path), mask)
            return

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                class_id = int(parts[0])
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                coords = coords.astype(np.int32)

                fill_value = 255 if self.num_classes == 1 else (class_id + 1)
                cv2.fillPoly(mask, [coords], fill_value)

        cv2.imwrite(str(output_path), mask)

    def convert_dataset(self, dataset_type='train'):
        """转换单个数据集（train/val/test）"""
        img_dir = self.base_path / self.data_config[dataset_type]
        label_dir = Path(str(img_dir).replace('images', 'labels'))
        mask_dir = Path(str(img_dir).replace('images', 'masks'))
        mask_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            label_path = label_dir / (img_path.stem + '.txt')
            mask_path = mask_dir / (img_path.stem + '.png')
            self.convert_single_label(label_path, (h, w), mask_path)

    def convert_all(self):
        """批量转换 train / val / test"""
        for dataset_type in ['train', 'val', 'test']:
            if dataset_type in self.data_config:
                self.convert_dataset(dataset_type)


def main():
    data_yaml = 'yolo11n_seg_data.yaml'
    if not os.path.exists(data_yaml):
        print(f"找不到配置文件: {data_yaml}")
        return

    converter = YOLOToSegNetConverter(data_yaml)
    converter.convert_all()


if __name__ == '__main__':
    main()
