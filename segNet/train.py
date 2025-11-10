import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from .timeT import TimeTracker




class SegNet(nn.Module):
    """SegNet模型实现"""
    def __init__(self, num_classes=1, in_channels=3):
        super(SegNet, self).__init__()
        
        # Encoder
        self.enc1 = self._encoder_block(in_channels, 64, 2)
        self.enc2 = self._encoder_block(64, 128, 2)
        self.enc3 = self._encoder_block(128, 256, 3)
        self.enc4 = self._encoder_block(256, 512, 3)
        self.enc5 = self._encoder_block(512, 512, 3)
        
        # Decoder
        self.dec5 = self._decoder_block(512, 512, 3)
        self.dec4 = self._decoder_block(512, 256, 3)
        self.dec3 = self._decoder_block(256, 128, 3)
        self.dec2 = self._decoder_block(128, 64, 2)
        self.dec1 = self._decoder_block(64, num_classes, 2)
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
    def _encoder_block(self, in_ch, out_ch, num_conv):
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_ch, out_ch, num_conv):
        layers = []
        for i in range(num_conv):
            conv_in = in_ch if i == 0 else out_ch
            layers.append(nn.Conv2d(conv_in, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            if i < num_conv - 1:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1_pool, idx1 = self.pool(e1)
        
        e2 = self.enc2(e1_pool)
        e2_pool, idx2 = self.pool(e2)
        
        e3 = self.enc3(e2_pool)
        e3_pool, idx3 = self.pool(e3)
        
        e4 = self.enc4(e3_pool)
        e4_pool, idx4 = self.pool(e4)
        
        e5 = self.enc5(e4_pool)
        e5_pool, idx5 = self.pool(e5)
        
        # Decoder
        d5 = self.unpool(e5_pool, idx5, output_size=e5.size())
        d5 = self.dec5(d5)
        
        d4 = self.unpool(d5, idx4, output_size=e4.size())
        d4 = self.dec4(d4)
        
        d3 = self.unpool(d4, idx3, output_size=e3.size())
        d3 = self.dec3(d3)
        
        d2 = self.unpool(d3, idx2, output_size=e2.size())
        d2 = self.dec2(d2)
        
        d1 = self.unpool(d2, idx1, output_size=e1.size())
        d1 = self.dec1(d1)
        
        return d1


class WaterSegDataset(Dataset):
    """水面分割数据集 - 直接读取转换后的mask"""
    def __init__(self, image_dir, mask_dir, img_size=640, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 获取所有图像文件
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                  list(self.image_dir.glob('*.png')))
        
        print(f"数据集加载: {len(self.image_files)} 张图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取对应的mask
        mask_path = self.mask_dir / (img_path.stem + '.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            # 如果mask不存在，创建空白mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 数据增强
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # 调整大小
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # 归一化mask到[0,1]
        mask = mask / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
    
    def _augment(self, image, mask):
        """简单的数据增强"""
        # 随机水平翻转
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # 随机亮度调整
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image, mask


class Trainer:
    """SegNet训练器"""
    def __init__(self, model, train_loader, val_loader, device, save_dir='runs/segnet'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
        self.best_loss = float('inf')
    
    def train_epoch(self, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self, criterion):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                # 计算IoU
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (pred_mask * masks).sum(dim=(1,2,3))
                union = pred_mask.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) - intersection
                iou = (intersection / (union + 1e-8)).mean().item()
                
                total_loss += loss.item()
                total_iou += iou
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        
        return avg_loss, avg_iou
    
    def save_checkpoint(self, epoch, optimizer, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        
        # 保存最后一个epoch
        torch.save(checkpoint, self.save_dir / 'last.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
    


def train_segnet(data_yaml='seg_data.yaml', 
                epochs=200, 
                batch_size=16, 
                img_size=640,
                lr=0.001, 
                device='cuda',
                workers=8):
    """主训练函数"""
    
    # 读取数据配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    base_path = Path(data_config['path'])
    
    # 获取图像和mask目录
    train_img_dir = base_path / data_config['train']
    train_mask_dir = Path(str(train_img_dir).replace('images', 'masks'))
    val_img_dir = base_path / data_config['val']
    val_mask_dir = Path(str(val_img_dir).replace('images', 'masks'))
    
    print("\n数据集路径:")
    print(f"  训练图像: {train_img_dir}")
    print(f"  训练Mask: {train_mask_dir}")
    print(f"  验证图像: {val_img_dir}")
    print(f"  验证Mask: {val_mask_dir}")
    
    # 检查mask是否存在
    if not train_mask_dir.exists():
        print(f"\n错误: 找不到训练集mask目录: {train_mask_dir}")
        print("请先运行 yolo_to_segnet_converter.py 转换数据格式！")
        return None
    
    # 创建数据集
    train_dataset = WaterSegDataset(train_img_dir, train_mask_dir, img_size, augment=True)
    val_dataset = WaterSegDataset(val_img_dir, val_mask_dir, img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=workers, pin_memory=True)
    
    # 创建模型
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model = SegNet(num_classes=1, in_channels=3).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # 训练循环
    print(f"\n开始训练 {epochs} 个epochs...")
    print("="*80)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*80)
        
        # 训练
        train_loss = trainer.train_epoch(optimizer, criterion)
        
        # 验证
        val_loss, val_iou = trainer.validate(criterion)
        
        # 记录历史
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_iou'].append(val_iou)
        
        # 打印结果
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # 保存最佳模型
        is_best = val_loss < trainer.best_loss
        if is_best:
            trainer.best_loss = val_loss
            print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        trainer.save_checkpoint(epoch, optimizer, is_best)
        
        # 更新学习率
        scheduler.step()
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("="*80)
    
    
    return model


if __name__ == '__main__':
    # 创建时间跟踪器
    time_tracker = TimeTracker()
    
    # 训练前记录时间
    time_tracker.on_train_start()
    
    # 训练SegNet模型
    model = train_segnet(
        data_yaml='seg_data.yaml',  # 数据集配置文件
        epochs=30,                          # 训练轮次
        batch_size=4,                       # 批量大小
        img_size=640,                        # 图像尺寸
        lr=0.001,                           # 学习率
        device='cuda',                       # 'cuda' 或 'cpu'
        workers=8                       # 数据加载线程数
    )
    
    # 训练后记录时间
    time_tracker.on_train_end()
    
    if model is not None:
        print("\n" + "="*80)
        print("训练完成！")
        print(f"最佳模型保存在: runs/segnet/best.pth")
        print(f"最新模型保存在: runs/segnet/last.pth")
        print("="*80)