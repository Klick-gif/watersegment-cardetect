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