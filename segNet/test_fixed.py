import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from .train import SegNet

def safe_torch_load(file_path, device='cpu'):
    """安全加载PyTorch模型文件，兼容PyTorch 2.6"""
    try:
        # 首先尝试weights_only=True（安全模式）
        return torch.load(file_path, weights_only=True, map_location=device)
    except Exception as e:
        print(f"安全模式加载失败: {e}，尝试非安全模式")
        # 如果安全模式失败，使用非安全模式（仅对可信文件使用）
        return torch.load(file_path, weights_only=False, map_location=device)

class WaterSegmentationPredictor:
    def __init__(self, model_path, img_size=640, device='cpu'):
        self.img_size = img_size
        # 强制使用CPU，避免GPU相关问题
        self.device = torch.device('cpu')
        print(f"使用设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        
        # 加载模型结构
        self.model = SegNet(num_classes=1, in_channels=3)
        
        try:
            # 使用安全的模型加载方式
            print(f"正在加载模型: {model_path}")
            checkpoint = safe_torch_load(model_path, self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("加载model_state_dict格式的权重")
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print("加载state_dict格式的权重")
            else:
                # 直接加载权重
                self.model.load_state_dict(checkpoint)
                print("直接加载权重")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"模型加载成功: {model_path}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 尝试其他加载方式
            try:
                print("尝试备用加载方式...")
                # 如果上面失败，尝试更宽松的加载方式
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # 尝试不同的键名
                possible_keys = ['model_state_dict', 'state_dict', 'model', 'weights']
                loaded = False
                
                for key in possible_keys:
                    if key in checkpoint:
                        self.model.load_state_dict(checkpoint[key])
                        print(f"使用键 '{key}' 加载权重")
                        loaded = True
                        break
                
                if not loaded:
                    # 直接尝试加载
                    self.model.load_state_dict(checkpoint)
                    print("直接加载整个checkpoint")
                
                self.model.to(self.device)
                self.model.eval()
                print(f"模型加载成功（备用方式）: {model_path}")
                
            except Exception as e2:
                print(f"所有加载方式都失败: {e2}")
                raise e2
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果已经是numpy数组
            image = image_path
        
        # 保存原始尺寸
        original_size = image.shape[:2]  # (height, width)
        
        # 调整大小并归一化
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # 添加batch维度
        
        return image_tensor, image, original_size
    
    def predict(self, image_path, confidence_threshold):
        """进行预测"""
        try:
            # 预处理
            image_tensor, original_image, original_size = self.preprocess_image(image_path)
            
            # 预测
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = torch.sigmoid(output)
                prediction = (probability > confidence_threshold).float()
            
            # 后处理
            mask = prediction.squeeze().cpu().numpy()  # 移除batch和channel维度
            mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))  # 调整回原始尺寸
            
            return mask_resized, original_image, probability.mean().item()
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 返回默认结果
            if isinstance(image_path, str):
                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            else:
                original_image = image_path
            
            mask = np.zeros(original_image.shape[:2], dtype=np.float32)
            return mask, original_image, 0.0
    
    def visualize(self, image_path, confidence_threshold=0.5, save_path=None):
        """可视化预测结果"""
        # 进行预测
        mask, original_image, prob = self.predict(image_path, confidence_threshold)
        
        # 创建彩色mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = [0, 255, 0]  # 绿色表示水面
        
        # 创建叠加图像
        overlay = original_image.copy()
        overlay[mask > 0] = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)[mask > 0]
        
        # 绘制结果
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 分割mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'水面分割Mask (置信度: {prob:.3f})')
        axes[1].axis('off')
        
        # 叠加结果
        axes[2].imshow(overlay)
        axes[2].set_title('分割结果叠加')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存至: {save_path}")
        
        plt.show()
        
        return mask, overlay
    
    def batch_predict(self, image_paths, confidence_threshold=0.5):
        """批量预测"""
        results = []
        for image_path in image_paths:
            mask, original_image, prob = self.predict(image_path, confidence_threshold)
            results.append({
                'mask': mask,
                'original_image': original_image,
                'confidence': prob,
                'water_pixels': np.sum(mask > 0),
                'total_pixels': mask.size,
                'water_ratio': np.sum(mask > 0) / mask.size
            })
        return results

def main():
    # 初始化预测器
    predictor = WaterSegmentationPredictor(
        model_path='segment/best.pth',  # 使用正确的模型路径
        img_size=640,
        device='cpu'
    )

    # 单张图像预测
    image_path = 'test_image.jpg'

    print("进行水面分割预测...")
    mask, overlay, confidence = predictor.predict(image_path, confidence_threshold=0.5)

    # 可视化结果
    predictor.visualize(image_path, confidence_threshold=0.5, save_path='prediction_result.png')

    # 打印统计信息
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