import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
from timeT import TimeTracker
import logging

# 禁用transformers的网络请求
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 禁用不必要的日志
logging.getLogger("transformers").setLevel(logging.ERROR)

def validate_detr(split_set, annotations_file):
    # 配置参数
    model_path = "detr_vehicle_flood_epoch_15"
    images_base_dir = "../YOLO11/yolo11n_data"
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DetrImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = DetrForObjectDetection.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()

    # 加载测试数据和类别信息
    with open(annotations_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 获取类别映射
    categories = test_data['categories']
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    category_ids = sorted([cat['id'] for cat in categories])
    
    # 根据split_set过滤图像
    split_images = [img for img in test_data['images'] if img.get('split') == split_set]
    
    # 初始化统计变量
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_correct = 0
    total_predictions = 0
    
    # 为每个类别初始化统计
    class_stats = {}
    for cat_id in category_ids:
        class_stats[cat_id] = {
            'name': category_id_to_name[cat_id],
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'predictions': 0
        }
    
    print(f"开始验证 {split_set} 集...")
    timetracker = TimeTracker()
    timetracker.on_train_start()
    
    for img_info in tqdm(split_images):
        image_path = os.path.join(images_base_dir, split_set, 'images', img_info['file_name'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            continue
        
        # 获取真实标注
        gt_annotations = [ann for ann in test_data['annotations'] if ann['image_id'] == img_info['id']]
        gt_boxes = []
        gt_labels = []
        
        for ann in gt_annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            # 转换为 [x_min, y_min, x_max, y_max]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])
        
        # 模型预测
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(device)  # [height, width]
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.6
        )[0]
        
        pred_boxes = results['boxes'].cpu().numpy()
        pred_scores = results['scores'].cpu().numpy()
        pred_labels = results['labels'].cpu().numpy()
        
        # 计算TP, FP, FN (基于IoU=0.5)
        used_gt_indices = set()
        
        for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0
            best_gt_idx = -1
            
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in used_gt_indices:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou and pred_label == gt_label:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= 0.5 and best_gt_idx != -1:
                total_tp += 1
                total_correct += 1
                class_stats[pred_label]['tp'] += 1
                used_gt_indices.add(best_gt_idx)
            else:
                total_fp += 1
                class_stats[pred_label]['fp'] += 1
            
            total_predictions += 1
            class_stats[pred_label]['predictions'] += 1
        
        # 未匹配的真实框是FN
        total_fn += len(gt_boxes) - len(used_gt_indices)
        
        # 为每个未匹配的GT框统计FN
        for j, gt_label in enumerate(gt_labels):
            if j not in used_gt_indices:
                class_stats[gt_label]['fn'] += 1
    
    # 计算总体指标
    accuracy = total_correct / (total_predictions + 1e-16)
    precision = total_tp / (total_tp + total_fp + 1e-16)
    recall = total_tp / (total_tp + total_fn + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    
    # 计算每个类别的指标
    class_results = []
    for cat_id in category_ids:
        stats = class_stats[cat_id]
        class_name = stats['name']
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        class_precision = tp / (tp + fp + 1e-16)
        class_recall = tp / (tp + fn + 1e-16)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-16)
        
        class_results.append({
            'name': class_name,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # 打印结果
    timetracker.on_train_end()
    print(f'{split_set}:')
    print(f'  精确率: {precision:.4f}')
    print(f'  准确率: {accuracy:.4f}')
    print(f'  召回率: {recall:.4f}')
    print(f'  F1分数: {f1:.4f}')
    print(f'  混淆矩阵 - TN:0, FP:{total_fp}, FN:{total_fn}, TP:{total_tp}')
    print()
    
    # 打印每个类别的详细指标
    print(f'  {split_set}集各类别详细指标:')
    print('  ' + '-' * 60)
    print(f'  {"类别名称":<15} {"精确率":<8} {"召回率":<8} {"F1分数":<8} {"TP":<4} {"FP":<4} {"FN":<4}')
    print('  ' + '-' * 60)
    
    for result in class_results:
        print(f'  {result["name"]:<15} {result["precision"]:.4f}  {result["recall"]:.4f}  {result["f1"]:.4f}  {result["tp"]:<4} {result["fp"]:<4} {result["fn"]:<4}')
    
    print('  ' + '-' * 60)
    print()

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height
    
    # 计算并集区域
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-16)

if __name__ == "__main__":
    # 分别验证train、val、test三个数据集
    test_annotations_file = "coco_format_data/test_annotations.json"
    train_annotations_file = "coco_format_data/train_annotations.json"
    val_annotations_file = "coco_format_data/val_annotations.json"
    
    # 创建映射字典
    annotations_files = {
        'train': train_annotations_file,
        'val': val_annotations_file,
        'test': test_annotations_file
    }
    
    for split in ['train', 'val', 'test']:
        validate_detr(split, annotations_file=annotations_files[split])