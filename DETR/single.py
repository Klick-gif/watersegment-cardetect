import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def predict_single_image():
    # 配置参数
    model_path = "detr_vehicle_flood_epoch_15"  # 本地训练好的模型目录
    image_path = "test_image.jpg"  # 替换为您的测试图片路径
    output_path = "prediction_result.jpg"  # 输出图片路径
    
    # 类别名称（根据您的数据）
    class_names = ['cc', 'cm', 'lt']
    
    # 加载本地模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        processor = DetrImageProcessor.from_pretrained(model_path, local_files_only=True)
        model = DetrForObjectDetection.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"成功加载本地模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 加载并处理图像
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"图像尺寸: {image.size}")
    except Exception as e:
        print(f"无法加载图像: {e}")
        return
    
    # 模型预测
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
    
    # 后处理
    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # [height, width]
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5  # 置信度阈值
    )[0]
    
    # 解析结果
    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()
    
    print(f"\n检测到 {len(scores)} 个目标:")
    print("-" * 50)
    
    # 在图像上绘制检测结果
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 颜色列表
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # if score < 0.5:  # 只显示置信度大于0.5的检测结果
        #     continue
            
        # 边界框坐标
        x_min, y_min, x_max, y_max = box
        
        # 类别信息
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
        confidence = score
        
        # 打印检测结果
        print(f"目标 {i+1}: {class_name} - 置信度: {confidence:.3f}")
        print(f"  位置: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
        
        # 绘制边界框
        color = colors[label % len(colors)]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        
        # 绘制标签背景
        label_text = f"{class_name}: {confidence:.2f}"
        try:
            bbox = draw.textbbox((x_min, y_min), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x_min, y_min), label_text, fill="white", font=font)
        except:
            # 如果字体绘制失败，使用简单文本
            draw.text((x_min, y_min), label_text, fill=color)
    
    # 保存结果图像
    image.save(output_path)
    print(f"\n结果已保存到: {output_path}")
    
    # 显示统计信息
    high_conf_detections = sum(scores >= 0.5)
    print(f"\n统计信息:")
    print(f"总检测数: {len(scores)}")
    print(f"高置信度检测数(>0.5): {high_conf_detections}")
    
    return image

if __name__ == "__main__":
    predict_single_image()