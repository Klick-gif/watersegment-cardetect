# water_segmentation_cpu.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from .train import SegNet      # 确保 train.py 里 SegNet 定义能 import 到


class WaterSegmentationPredictor:
    def __init__(self, model_path, img_size=640, device='cpu'):
        self.img_size = img_size
        # 强制 CPU
        self.device = torch.device('cpu')
        print(f"使用设备: {self.device}")

        # 加载模型结构
        self.model = SegNet(num_classes=1, in_channels=3)

        # 加载训练好的权重（统一 map_location 到 CPU）
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:   # 整个文件就是 state_dict
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载成功: {model_path}")

    # ---------- 以下逻辑与 GPU 版完全一致 ----------
    def preprocess_image(self, image_path):
        """预处理图像"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:               # 传入的是 numpy 数组
            image = image_path

        original_size = image.shape[:2]          # (H, W)
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)   # (1,3,H,W)
        return image_tensor, image, original_size

    def predict(self, image_path, confidence_threshold=0.5):
        """单张预测"""
        image_tensor, original_image, original_size = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)          # (1,1,H,W)
            probability = torch.sigmoid(output)
            prediction = (probability > confidence_threshold).float()

        mask = prediction.squeeze().cpu().numpy()      # (H,W)
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
        return mask_resized, original_image, probability.mean().item()

    def visualize(self, image_path, confidence_threshold=0.5, save_path=None):
        """可视化"""
        mask, original_image, prob = self.predict(image_path, confidence_threshold)

        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = [0, 255, 0]      # 绿色

        overlay = original_image.copy()
        overlay[mask > 0] = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)[mask > 0]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_image);   axes[0].set_title('原始图像'); axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray');axes[1].set_title(f'水面Mask (置信度:{prob:.3f})');axes[1].axis('off')
        axes[2].imshow(overlay);          axes[2].set_title('分割叠加'); axes[2].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存至: {save_path}")
        plt.show()
        return mask, overlay

    def batch_predict(self, image_paths, confidence_threshold=0.5):
        """批量预测"""
        results = []
        for path in image_paths:
            mask, original_image, prob = self.predict(path, confidence_threshold)
            results.append({
                'mask': mask,
                'original_image': original_image,
                'confidence': prob,
                'water_pixels': np.sum(mask > 0),
                'total_pixels': mask.size,
                'water_ratio': np.sum(mask > 0) / mask.size
            })
        return results


# ------------------- demo -------------------
def main():
    predictor = WaterSegmentationPredictor(
        model_path='runs/segnet/best.pth',   # 你的权重
        img_size=640,
        device='cpu'                         # 显式指定 CPU
    )

    image_path = 'test_image.jpg'            # 测试图
    print("进行水面分割预测（CPU）...")
    mask, overlay, confidence = predictor.predict(image_path, confidence_threshold=0.5)

    predictor.visualize(image_path, confidence_threshold=0.5,
                        save_path='prediction_result_cpu.png')

    # 统计
    water_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    water_ratio = water_pixels / total_pixels
    print(f"\n预测结果统计:")
    print(f"  水面像素数: {water_pixels}")
    print(f"  总像素数: {total_pixels}")
    print(f"  水面比例: {water_ratio:.2%}")
    print(f"  模型置信度: {confidence:.3f}")


if __name__ == "__main__":
    main()