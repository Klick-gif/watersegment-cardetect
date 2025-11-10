import warnings
import multiprocessing
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from timeT import TimeTracker
import numpy as np


def evaluate_segmentation(model_path, data_yaml, split_set='val'):
    """
    评价YOLO分割模型
    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件
        split_set: 数据集类型 'train', 'val', 'test'
    """
    # 加载模型
    model = YOLO(model_path)
    time_tracker = TimeTracker()
    time_tracker.on_train_start()
    
    # 执行分割验证
    metrics = model.val(
        data=data_yaml,
        split=split_set,
        imgsz=640,
        batch=16,
        iou=0.60,
        conf=0.001,
        workers=0,
        task='segment'
    )
    time_tracker.on_train_end()
    
    # 仿照检测模型的输出格式
    print(f'{split_set}:')
    print(f'  mAP50  : {metrics.box.map50:.4f}')
    print(f'  mAP50-95: {metrics.box.map:.4f}')
    print(f'  精确率: {metrics.box.mp:.4f}')
    print(f'  召回率: {metrics.box.mr:.4f}')
    
    # 计算F1分数
    p, r = metrics.box.mp, metrics.box.mr
    f1 = 2 * p * r / (p + r + 1e-16)
    print(f'  F1分数: {f1:.4f}')
    
    # 获取并分析混淆矩阵
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        cm = metrics.confusion_matrix.matrix
        
        # 假设是二分类情况，提取TN, FP, FN, TP
        if cm.shape == (2, 2):  # 二分类
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        else:  # 多分类，需要汇总
            # 对于多分类，我们需要计算总的TN, FP, FN, TP
            # 这里假设最后一个维度是背景类
            num_classes = cm.shape[0] - 1  # 假设最后一个类是背景
            
            TP = np.sum(np.diag(cm[:num_classes, :num_classes]))
            FP = np.sum(cm[:num_classes, :num_classes]) - TP
            FN = np.sum(cm[num_classes, :num_classes])
            TN = cm[num_classes, num_classes]
        
        print(f'  混淆矩阵 - TN:{int(TN)}, FP:{int(FP)}, FN:{int(FN)}, TP:{int(TP)}')
        
        # 计算准确率
        total = TN + FP + FN + TP
        accuracy = (TP + TN) / total if total > 0 else 0
        print(f'  准确率: {accuracy:.4f}')
    else:
        print('  混淆矩阵: 不可用')
    
    print()  # 空行分隔不同数据集的结果
    
    return metrics


def main():
    # 模型路径 - 分割模型路径
    model_path = 'runs/segment/train/weights/best.pt'
    
    # 数据集配置
    data_yaml = 'yolo11n_seg_data.yaml'
    
    # 分别评价三个数据集
    for split_set in ['train', 'val', 'test']:
        print(f"--- 正在评估 {split_set} 数据集 ---")
        
        try:
            evaluate_segmentation(model_path, data_yaml, split_set)
            
        except Exception as e:
            print(f"评估 {split_set} 数据集时出错: {e}\n")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()