import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import time
from datetime import datetime

class TimeTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def on_train_start(self):
        self.start_time = time.time()
        print(f"🚀 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_train_end(self):
        self.end_time = time.time()
        training_time = self.end_time - self.start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        print(f"✅ 训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ 总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == '__main__':
    # 创建时间跟踪器
    time_tracker = TimeTracker()
    model = YOLO('ultralytics/cfg/models/11/yolo11n-seg.yaml')
    model.load('分割预训练模型/yolo11n-seg.pt')  # 注释则不加载
    # 训练前记录时间
    time_tracker.on_train_start()
    results = model.train(
        data='yolo11n_seg_data.yaml',  # 数据集配置文件的路径
        epochs=200,  # 训练轮次总数
        batch=16,  # 批量大小，即单次输入多少图片训练
        imgsz=640,  # 训练图像尺寸
        workers=8,  # 加载数据的工作线程数
        device=0,  # 指定训练的计算设备，无 nvidia 显卡则改为 'cpu'
        optimizer='SGD',  # 训练使用优化器，可选 auto,SGD,Adam,AdamW 等
        amp=True,  # True 或者 False，解释为：自动混合精度(AMP)训练
        cache=False  # True 在内存中缓存数据集图像，服务器推荐开启
    )
    # 训练后记录时间
    time_tracker.on_train_end()