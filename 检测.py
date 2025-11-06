import torch

# 1. 检查CUDA（GPU）是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. 检查当前PyTorch使用的CUDA版本
print(f"PyTorch CUDA version: {torch.version.cuda}")

# 3. 检查当前PyTorch使用的cuDNN版本
print(f"cuDNN version: {torch.backends.cudnn.version()}")

# 4. 查看你的GPU型号
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")