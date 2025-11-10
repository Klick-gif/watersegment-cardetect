import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class FloodDetectionDataset(Dataset):
    def __init__(self, images_base_dir, annotations_file, processor, split='train'):
        self.images_base_dir = images_base_dir
        self.processor = processor
        self.split = split
        
        # 加载 COCO JSON
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            self.image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)
        
        # 按 split 过滤
        self.image_ids = [
            img['id'] for img in self.coco_data['images']
        ]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]
        image_path = os.path.join(self.images_base_dir, self.split, 'images', image_info['file_name'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"图像加载失败: {image_path}, 错误: {e}")
            return None
        
        annotations = self.image_id_to_annotations.get(image_id, [])
        coco_annotations = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                coco_annotations.append({
                    'id': ann['id'],
                    'image_id': image_id,
                    'category_id': ann['category_id'],
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': 0
                })
        
        target = {'image_id': image_id, 'annotations': coco_annotations}
        
        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )
        
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_mask = encoding["pixel_mask"].squeeze()
        labels = encoding["labels"][0] if "labels" in encoding else {
            "class_labels": torch.zeros(0, dtype=torch.int64),
            "boxes": torch.zeros(0, 4, dtype=torch.float32)
        }
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels
    }
