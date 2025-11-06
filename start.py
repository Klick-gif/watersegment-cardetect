import uvicorn
import os
import sys

def check_requirements():
    """检查依赖是否安装"""
    try:
        import fastapi
        import ultralytics
        import cv2
        import PIL
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_models():
    """检查模型文件是否存在"""
    models_exist = True
    
    if not os.path.exists("detect/yolo11_best.pt"):
        print("⚠️  检测模型文件不存在: detect/yolo11_best.pt")
        models_exist = False
    
    if not os.path.exists("segment/yolo11_best.pt"):
        print("⚠️  分割模型文件不存在: segment/yolo11_best.pt")
        models_exist = False
    
    if models_exist:
        print("✅ 模型文件检查完成")
    else:
        print("⚠️  部分模型文件缺失，系统仍可运行但相关功能可能不可用")
    
    return models_exist

def main():
    """主函数"""
    print("积水识别和车辆淹没部位判别系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_requirements():
        sys.exit(1)
    
    # 检查模型
    check_models()
    
    # 创建必要目录
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print("\n🚀 启动服务器...")
    print("📱 访问地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    print("🛑 按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # 添加这一行
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
