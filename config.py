import os

# 环境配置
class Config:
    """应用配置类"""
    
    # 基础配置
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # 模型配置
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '-1')
    
    # 文件路径
    UPLOAD_FOLDER = 'uploads'
    RESULT_FOLDER = 'results'
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    # 性能优化（CPU专用）
    BATCH_SIZE = 1  # CPU环境下使用较小的批次
    NUM_WORKERS = 0  # CPU环境下不使用多线程加载
    
    # 模型路径
    MODEL_PATHS = {
        "detect": "detect/yolo11_best.pt",
        "detr": "detect/detr_recnet50",
        "segment": "segment/yolo11_best.pt",
        "segnet": "segment/best.pth"
    }

# 创建配置实例
config = Config()