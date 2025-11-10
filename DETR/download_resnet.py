#!/usr/bin/env python3
"""
手动下载 ResNet50 预训练权重
"""
import os
import requests
from pathlib import Path

def download_resnet_weights():
    """下载 ResNet50 预训练权重到本地缓存"""
    
    # 创建缓存目录
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # ResNet50 权重 URL
    resnet_url = "https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/model.safetensors"
    
    # 目标文件路径
    target_file = cache_dir / "models--timm--resnet50.a1_in1k" / "snapshots" / "main" / "model.safetensors"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在下载 ResNet50 权重到: {target_file}")
    
    try:
        # 下载文件
        response = requests.get(resnet_url, stream=True)
        response.raise_for_status()
        
        with open(target_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ ResNet50 权重下载完成: {target_file}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    download_resnet_weights()
