# YOLO11-
基于Yolo11基础模型进行微调，得到的目标检测模型以及水面分割模型

# 🌊 积水识别和车辆淹没部位判别系统

基于FastAPI和YOLO11的智能图像分析平台，用于识别积水场景中的车辆淹没部位。

## ✨ 功能特性

- 🖼️ **图像上传**: 支持拖拽上传和点击选择图片
- 🎯 **目标检测**: 基于YOLO11的高精度目标检测
- 🔍 **图像分割**: 精确的像素级分割分析
- 📊 **统计分析**: 车辆淹没部位数量统计
- 🔄 **模型切换**: 支持检测和分割模型切换
- 📥 **结果下载**: 支持检测结果和原始图片下载
- 🖥️ **左右分屏**: 原图与检测结果对比显示
- 📱 **响应式设计**: 适配各种屏幕尺寸

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件

将您的YOLO11模型文件放置在以下位置：
- `detect/yolo11_best.pt` - 目标检测模型
- `segment/yolo11_best.pt` - 图像分割模型

### 3. 启动系统

```bash
python start.py
```

或者直接运行：

```bash
python main.py
```

### 4. 访问系统

打开浏览器访问：http://localhost:8000

## 📁 项目结构

```
app/
├── main.py                 # FastAPI主程序
├── start.py                # 启动脚本
├── requirements.txt        # 依赖文件
├── README.md              # 说明文档
├── detect/                # 检测模型目录
│   └── yolo11_best.pt     # 检测模型权重
├── segment/               # 分割模型目录
│   └── yolo11_best.pt     # 分割模型权重
├── templates/             # 前端模板
│   └── index.html         # 主页面
├── uploads/               # 上传文件目录
├── results/               # 结果输出目录
└── static/                # 静态文件目录
```

## 🔧 API接口

### 上传图片
```http
POST /upload
Content-Type: multipart/form-data

file: 图片文件
```

### 图像预测
```http
POST /predict
Content-Type: application/json

{
    "file_id": "文件ID",
    "model_type": "detect|segment",
    "confidence": 0.25
}
```

### 下载结果
```http
GET /download/{file_id}/{model_type}
GET /download_original/{file_id}
```

## 🎨 界面功能

### 主界面
- **上传区域**: 支持拖拽和点击上传图片
- **控制面板**: 模型选择、置信度调节、预测按钮
- **结果显示**: 左右分屏显示原图和检测结果
- **统计信息**: 车辆淹没部位数量统计
- **检测详情**: 详细的检测结果列表
- **下载功能**: 支持结果图片和原始图片下载

### 车辆淹没部位类别
- 车轮
- 车身  
- 车顶
- 引擎盖
- 后备箱

## ⚙️ 配置说明

### 模型配置
在 `main.py` 中的 `MODELS` 字典中配置模型：

```python
MODELS = {
    "detect": {
        "name": "目标检测模型",
        "path": "detect/yolo11_best.pt",
        "type": "detect"
    },
    "segment": {
        "name": "图像分割模型", 
        "path": "segment/yolo11_best.pt",
        "type": "segment"
    }
}
```

### 车辆部位映射
在 `VEHICLE_PARTS` 字典中配置类别映射：

```python
VEHICLE_PARTS = {
    0: "车窗",
    1: "车门",
    2: "车轮",
}
```

## 🛠️ 技术栈

- **后端**: FastAPI, Uvicorn
- **AI模型**: YOLO11 (Ultralytics)
- **图像处理**: OpenCV, Pillow
- **前端**: HTML5, CSS3, JavaScript
- **部署**: Python 3.8+

## 📝 使用说明

1. **上传图片**: 点击上传区域或拖拽图片到指定区域
2. **选择模型**: 在控制面板中选择检测或分割模型
3. **调节参数**: 调整置信度阈值（0.1-0.9）
4. **开始预测**: 点击"开始预测"按钮
5. **查看结果**: 在左右分屏中查看原图和检测结果
6. **下载结果**: 点击下载按钮保存结果图片

## 🔍 故障排除

### 常见问题

1. **模型文件不存在**
   - 确保模型文件路径正确
   - 检查文件权限

2. **依赖安装失败**
   - 使用虚拟环境
   - 升级pip版本

3. **端口被占用**
   - 修改 `main.py` 中的端口号
   - 检查防火墙设置

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 支持

如有问题，请查看API文档：http://localhost:8000/docs
