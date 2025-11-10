import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import get_scheduler
import torch.optim as optim
from tqdm.auto import tqdm
import os
from dataset import collate_fn, FloodDetectionDataset
from timeT import TimeTracker

time_tracker = TimeTracker()


def train_detr_model():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 类别信息（根据您的数据）
    num_classes = 3  # cc, cm, lt 三个类别
    
    # 数据路径
    yolo_data_path = "../YOLO11/yolo11n_data"
    coco_output_dir = "coco_format_data"
    
    # 加载processor，设置固定的图像尺寸
    processor = DetrImageProcessor.from_pretrained("detr-r50", size={"height": 640, "width": 640})
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = FloodDetectionDataset(
        images_base_dir=yolo_data_path,
        annotations_file=os.path.join(coco_output_dir, "train_annotations.json"),
        processor=processor,
        split='train'
    )
    
    val_dataset = FloodDetectionDataset(
        images_base_dir=yolo_data_path,
        annotations_file=os.path.join(coco_output_dir, "val_annotations.json"),
        processor=processor,
        split='val'
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=8,  # 减小batch_size避免内存问题
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2  # 先设置为0避免多进程问题
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=8,  # 验证时使用更小的batch_size
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # 加载预训练模型
    print("加载DETR模型...")
    model = DetrForObjectDetection.from_pretrained(
        "detr-r50",
        num_labels=num_classes + 1,  # 背景 + 实类
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    
    num_epochs = 30
    num_training_steps = num_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # 训练循环
    print("开始训练...")
    time_tracker.on_train_start()
    model.train()
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 5  # 早停耐心：连续多少 epoch 验证集没提升就停止
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        model.train()
        for batch_idx, batch in enumerate(progress_bar):
            try:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = batch["labels"]

                # 移动标签到设备
                device_labels = []
                for label in labels:
                    device_label = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in label.items()}
                    device_labels.append(device_label)

                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=device_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
            except Exception as e:
                print(f"训练过程中出错: {e}")
                continue

        torch.cuda.empty_cache()

        # 验证集评估
        avg_train_loss = total_loss / len(train_dataloader)
        val_loss = evaluate_model(model, val_dataloader, device)

        print(f"\nEpoch {epoch+1}/{num_epochs} | 训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f}")

        # 判断是否为最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve_count = 0

            save_path = f"detr_vehicle_flood_best"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"✅ 验证损失改进，保存最优模型到: {save_path}")
        else:
            no_improve_count += 1
            print(f"⚠️ 验证损失未提升，连续 {no_improve_count}/{patience} 次")

        # 早停机制
        if no_improve_count >= patience:
            print(f"\n⏹️ 早停触发：验证集 {patience} 次未提升。最佳模型在第 {best_epoch} 轮 (val_loss={best_val_loss:.4f})。")
            break

    # 训练结束
    print(f"\n🎯 训练完成！最佳模型出现在第 {best_epoch} 轮，验证损失为 {best_val_loss:.4f}")
    time_tracker.on_train_end()

def evaluate_model(model, dataloader, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证"):
            try:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = batch["labels"]
                
                # 将labels移动到设备
                device_labels = []
                for label in labels:
                    device_label = {}
                    for k, v in label.items():
                        if isinstance(v, torch.Tensor):
                            device_label[k] = v.to(device)
                        else:
                            device_label[k] = v
                    device_labels.append(device_label)
                
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=device_labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"验证过程中出错: {e}")
                continue
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0

if __name__ == "__main__":
    train_detr_model()