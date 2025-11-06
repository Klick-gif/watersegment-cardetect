from ultralytics import YOLO
import cv2


def predict_image(model_path, image_path, conf=0.25):
    """
    单张图片预测
    Args:
        model_path: 模型权重路径
        image_path: 输入图片路径
        conf: 置信度阈值
    Returns:
        results: 预测结果
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 进行预测
    results = model.predict(
        source=image_path,
        conf=conf,
        project='runs/detect',
        name='predict',
        exist_ok=True
    )
    
    return results

# 使用示例
if __name__ == '__main__':
    results = predict_image(
        model_path='detect/yolo11_best.pt',
        image_path='test_image.jpg',
        conf=0.25
    )
    
    # 打印检测结果
    for r in results:
        boxes = r.boxes
        print(f"检测到 {len(boxes)} 个目标")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"类别: {cls}, 置信度: {conf:.3f}")