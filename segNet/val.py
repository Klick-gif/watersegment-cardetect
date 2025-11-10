import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from pathlib import Path
import yaml
from sklearn.metrics import confusion_matrix

class SegNetEvaluator:
    """SegNet模型评估器"""
    
    def __init__(self, model_path, data_yaml='seg_data.yaml', img_size=640, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        # 读取数据配置
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        base_path = Path(data_config['path'])
        
        # 获取数据集路径
        self.train_img_dir = base_path / data_config['train']
        self.train_mask_dir = Path(str(self.train_img_dir).replace('images', 'masks'))
        self.val_img_dir = base_path / data_config['val'] 
        self.val_mask_dir = Path(str(self.val_img_dir).replace('images', 'masks'))
        self.test_img_dir = base_path / data_config['test']
        self.test_mask_dir = Path(str(self.test_img_dir).replace('images', 'masks'))
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        from train import SegNet  # 导入你的模型定义
        
        model = SegNet(num_classes=1, in_channels=3)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✓ 模型加载成功: {model_path}")
        return model
    
    def evaluate_dataset(self, image_dir, mask_dir, dataset_name="数据集"):
        """评估单个数据集"""
        from train import WaterSegDataset  # 导入你的数据集定义
        
        dataset = WaterSegDataset(image_dir, mask_dir, self.img_size, augment=False)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        
        total_iou = 0.0
        total_pa = 0.0
        total_samples = 0
        
        # 用于混淆矩阵的统计量
        tn_total, fp_total, fn_total, tp_total = 0, 0, 0, 0
        
        print(f"\n正在评估 {dataset_name}...")
        
        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                
                # 计算IoU和PA
                batch_iou, batch_pa = self.calculate_metrics(pred_masks, masks)
                batch_size = images.size(0)
                
                total_iou += batch_iou * batch_size
                total_pa += batch_pa * batch_size
                total_samples += batch_size
                
                # 计算当前batch的混淆矩阵
                batch_tn, batch_fp, batch_fn, batch_tp = self.calculate_confusion_matrix(pred_masks, masks)
                tn_total += batch_tn
                fp_total += batch_fp
                fn_total += batch_fn
                tp_total += batch_tp
        
        # 计算平均指标
        mean_iou = total_iou / total_samples
        mean_pa = total_pa / total_samples
        
        print(f"{dataset_name}评估结果:")
        print(f"  样本数量: {total_samples}")
        print(f"  平均IoU: {mean_iou:.4f}")
        print(f"  平均PA: {mean_pa:.4f}")
        
        print(f"  混淆矩阵:")
        print(f"    TN: {tn_total}, FP: {fp_total}")
        print(f"    FN: {fn_total}, TP: {tp_total}")
        
        # 计算精确率和召回率
        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        
        return {
            'mean_iou': mean_iou,
            'mean_pa': mean_pa,
            'confusion_matrix': (tn_total, fp_total, fn_total, tp_total),
            'samples': total_samples,
            'precision': precision,
            'recall': recall
        }
    
    def calculate_metrics(self, pred, target):
        """计算IoU和PA指标"""
        # 将张量转换为numpy数组
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        batch_iou = 0.0
        batch_pa = 0.0
        
        for i in range(pred_np.shape[0]):
            # IoU计算
            intersection = np.logical_and(pred_np[i] > 0.5, target_np[i] > 0.5).sum()
            union = np.logical_or(pred_np[i] > 0.5, target_np[i] > 0.5).sum()
            iou = intersection / (union + 1e-8)
            batch_iou += iou
            
            # PA计算
            pa = (pred_np[i] == target_np[i]).sum() / target_np[i].size
            batch_pa += pa
        
        return batch_iou / pred_np.shape[0], batch_pa / pred_np.shape[0]
    
    def calculate_confusion_matrix(self, pred, target):
        """计算混淆矩阵 - 批量处理避免内存爆炸"""
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        # 使用二值化
        pred_binary = (pred_np > 0.5).astype(np.uint8)
        target_binary = (target_np > 0.5).astype(np.uint8)
        
        # 手动计算混淆矩阵
        tn = np.logical_and(pred_binary == 0, target_binary == 0).sum()
        fp = np.logical_and(pred_binary == 1, target_binary == 0).sum()
        fn = np.logical_and(pred_binary == 0, target_binary == 1).sum()
        tp = np.logical_and(pred_binary == 1, target_binary == 1).sum()
        
        return tn, fp, fn, tp
    
    def evaluate_all(self):
        """评估所有数据集"""
        print("=" * 60)
        print("SegNet模型评估")
        print("=" * 60)
        
        results = {}
        
        # 评估训练集
        if self.train_img_dir.exists() and self.train_mask_dir.exists():
            results['train'] = self.evaluate_dataset(self.train_img_dir, self.train_mask_dir, "训练集")
        else:
            print("⚠ 训练集路径不存在，跳过训练集评估")
        
        # 评估验证集
        if self.val_img_dir.exists() and self.val_mask_dir.exists():
            results['val'] = self.evaluate_dataset(self.val_img_dir, self.val_mask_dir, "验证集")
        else:
            print("⚠ 验证集路径不存在，跳过验证集评估")
        
        # 评估测试集
        if self.test_img_dir.exists() and self.test_mask_dir.exists():
            results['test'] = self.evaluate_dataset(self.test_img_dir, self.test_mask_dir, "测试集")
        else:
            print("⚠ 测试集路径不存在，跳过测试集评估")
        
        # 打印汇总结果
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """打印汇总结果"""
        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)
        
        for dataset_name, result in results.items():
            print(f"{dataset_name}:")
            print(f"  样本数: {result['samples']}")
            print(f"  mIoU: {result['mean_iou']:.4f}")
            print(f"  mPA:  {result['mean_pa']:.4f}")
            print(f"  精确率: {result['precision']:.4f}")
            print(f"  召回率: {result['recall']:.4f}")
            
            tn, fp, fn, tp = result['confusion_matrix']
            print(f"  混淆矩阵 - TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
            print()


def main():
    """主函数"""
    # 模型路径 - 根据需要修改
    model_path = "runs/segnet/best.pth"  # 或 "runs/segnet/last.pth"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练模型或检查模型路径！")
        return
    
    # 创建评估器
    evaluator = SegNetEvaluator(
        model_path=model_path,
        data_yaml='seg_data.yaml',
        img_size=640,
        device='cuda'
    )
    
    # 执行评估
    results = evaluator.evaluate_all()


if __name__ == '__main__':
    main()