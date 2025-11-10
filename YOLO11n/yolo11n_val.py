import warnings, multiprocessing
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from timeT import TimeTracker
import numpy as np

def main(split_set):
    model = YOLO(r'runs\\detect\\train4\\weights\\best.pt')
    time_tracker = TimeTracker()
    time_tracker.on_train_start()
    
    metrics = model.val(
        data='yolo11n_data.yaml',
        split=split_set,
        imgsz=640,
        batch=16,
        iou=0.50,
        conf=0.01,
        workers=0
    )
    time_tracker.on_train_end()
    
    # 获取数据集大小（需要从数据集中读取）
    # 由于YOLO的val方法不直接返回样本数，我们需要从数据集中获取
    
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
            # 这里假设最后一个维度是背景类（如果需要调整请根据实际情况修改）
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

if __name__ == '__main__':
    multiprocessing.freeze_support()   # Windows 必加
    for i in ['train', 'val', 'test']:
        main(i)