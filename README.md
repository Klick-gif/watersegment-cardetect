# 🌊 城市内涝监测系统 - 基于深度学习的积水识别和车辆淹没部位判别

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**基于YOLO11、DETR和SegNet的智能图像分析平台，用于城市内涝场景下的积水识别和车辆淹没部位自动判别**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [项目结构](#-项目结构) • [API文档](#-api接口) • [部署指南](#-部署方式)

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [模型准备](#模型准备)
  - [启动系统](#启动系统)
- [项目结构](#-项目结构)
- [模型介绍](#-模型介绍)
- [API接口](#-api接口)
- [部署方式](#-部署方式)
- [使用说明](#-使用说明)
- [配置说明](#-配置说明)
- [故障排除](#-故障排除)
- [技术栈](#-技术栈)
- [模型性能](#-模型性能)
- [开发计划](#-开发计划)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

- [模型微调指南](#-模型微调指南)
  - [YOLO11](#YOLO11)
  - [DETR](#DETR)
  - [segNet](#segNet)


---

## 📖 项目简介

城市内涝是指由于强降水或连续降水超过城市排水能力致使城市内产生积水灾害的现象。严重的城市洪涝往往会导致大量车辆被水浸泡淹没，造成车辆受损和人员被困。

本项目是一个基于图像和深度学习的**城市内涝监测系统**，主要功能包括：

1. **车辆淹没部位检测**：识别洪涝场景中车辆的淹没情况，判断车窗、车门、车轮等关键部位是否被水淹没，帮助评估车辆安全性和是否需要拖车避险。

2. **水面分割**：对积水区域进行精确分割，识别水面覆盖范围，为城市排水和应急管理提供数据支持。

### 项目背景

传统YOLO模型主要检测常见对象（如猫、狗、车、人等），而本项目针对城市内涝场景进行了专门优化和微调，能够：
- 检测车辆在积水环境中的淹没部位
- 精确分割积水水面区域
- 提供实时检测和分析能力
- 支持多种深度学习模型（YOLO11、DETR、SegNet）

![背景1](./image/背景1.png)
![背景2](./image/背景2.png)



---

## ✨ 功能特性

### 🎯 核心功能

- **🚗 多模型支持**：集成YOLO11、DETR、SegNet等多种深度学习模型
- **⚡ 实时检测**：支持水面分割、车辆部件检测、积水区域识别
- **📊 统计分析**：车辆淹没部位数量统计和详细检测结果展示
- **🌐 Web界面**：提供友好的Web界面进行图像上传、结果展示、模型切换
- **📱 响应式设计**：支持左右分屏对比显示、结果下载
- **🔌 RESTful API**：完整的API接口，支持第三方集成
- **💻 多环境支持**：支持CPU和GPU环境部署，提供优化配置
- **🐳 Docker部署**：支持容器化部署，一键启动

### 🎨 界面功能

- **上传区域**：支持拖拽和点击上传图片
- **控制面板**：模型选择、置信度调节、预测按钮
- **结果显示**：左右分屏显示原图和检测结果
- **统计信息**：车辆淹没部位数量统计
- **检测详情**：详细的检测结果列表
- **下载功能**：支持结果图片和原始图片下载

### 🚗 检测类别

系统支持以下车辆淹没部位检测：

- **车窗** (Window)
- **车门** (Door)
- **车轮** (Wheel)

---

## 🚀 快速开始

### 环境要求

| 组件     | 版本要求            | 说明                  |
| -------- | ------------------- | --------------------- |
| Python   | 3.10+               | 推荐使用Python 3.11.0 |
| PyTorch  | 2.7.1               | 支持CPU和GPU版本      |
| CUDA     | 11.8+               | 可选，GPU环境需要     |
| 操作系统 | Windows/Linux/macOS | 跨平台支持            |

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd <file>
```

#### 2. 安装依赖

**GPU环境（推荐）：**

```bash
pip install -r requirements.txt
```

**CPU环境：**

```bash
pip install -r requirements-cpu.txt
```

> **注意**：CPU版本使用CPU优化的PyTorch，适合无GPU环境的部署。

#### 3. 安装系统依赖（Linux）

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 fonts-wqy-microhei
```

### 模型准备

将训练好的模型文件放置在以下位置：

#### YOLO11模型
- `detect/yolo11_best.pt` - 目标检测模型
- `segment/yolo11_best.pt` - 图像分割模型

#### DETR模型
- `detect/detr_recnet50/` - DETR目标检测模型目录
  - `config.json`
  - `model.safetensors` 或 `pytorch_model.bin`
  - `preprocessor_config.json`

#### SegNet模型
- `segment/best.pth` - SegNet分割模型


### 启动系统

#### 方式一：标准启动（GPU环境）

```bash
python main.py
```

#### 方式二：CPU优化版本

```bash
python main-cpu.py
```

#### 方式三：使用启动脚本（推荐）

```bash
python start.py
```

启动脚本会自动：
- 检查依赖是否安装
- 检查模型文件是否存在
- 创建必要的目录
- 启动服务器

#### 方式四：Docker部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

详细部署指南请参考 [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md)

### 访问系统

启动成功后，打开浏览器访问：

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health (CPU版本)

---

## 📁 项目结构

```
app/
├── main.py                    # FastAPI主程序（GPU版本）
├── main-cpu.py                # FastAPI主程序（CPU优化版本）
├── start.py                   # 启动脚本
├── config.py                  # 配置文件
├── requirements.txt           # 依赖文件（GPU版本）
├── requirements-cpu.txt       # 依赖文件（CPU版本）
├── README.md                  # 项目说明文档
├── DEPLOYMENT-GUIDE.md        # 部署指南
├── Dockerfile                 # Docker镜像构建文件
├── docker-compose.yml         # Docker Compose配置
├── MSYH.TTC                   # 中文字体文件
│
├── detect/                    # 检测模型目录
│   ├── yolo11_best.pt        # YOLO11检测模型
│   └── detr_recnet50/        # DETR模型目录
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
│
├── segment/                   # 分割模型目录
│   ├── yolo11_best.pt        # YOLO11分割模型
│   └── best.pth               # SegNet分割模型
│
├── segNet/                    # SegNet模型代码
│   ├── train.py              # 训练脚本
│   ├── test.py               # 测试脚本
│   └── test_fixed.py         # 修复版测试脚本
│
├── templates/                 # 前端模板
│   └── index.html            # 主页面
│
├── uploads/                   # 上传文件目录
├── results/                   # 结果输出目录
└── static/                    # 静态文件目录
```

---

## 🔧 API接口

### 上传图片

**接口**: `POST /upload`

**请求格式**:
```http
POST /upload
Content-Type: multipart/form-data

file: <图片文件>
```

**响应示例**:
```json
{
  "file_id": "uuid-string",
  "filename": "uuid-string.jpg",
  "message": "图片上传成功"
}
```

### 图像预测

**接口**: `POST /predict`

**请求格式**:
```http
POST /predict
Content-Type: application/json

{
  "file_id": "文件ID",
  "model_type": "detect|detr|segment|segnet",
  "confidence": 0.60
}
```

**参数说明**:
- `file_id`: 上传文件后返回的文件ID
- `model_type`: 模型类型
  - `detect`: YOLO11检测模型
  - `detr`: DETR检测模型
  - `segment`: YOLO11分割模型
  - `segnet`: SegNet分割模型
- `confidence`: 置信度阈值（0.1-0.9）

**响应示例**:
```json
{
  "success": true,
  "file_id": "uuid-string",
  "model_type": "detect",
  "result_dir": "results/uuid-string_detect",
  "detections": [
    {
      "class_id": 0,
      "class_name": "车窗",
      "confidence": 0.85,
      "bbox": [100, 200, 300, 400]
    }
  ],
  "vehicle_stats": {
    "车窗": 2,
    "车门": 1,
    "车轮": 4
  },
  "result_image": "/results/uuid-string_detect/image.jpg",
  "total_detections": 7
}
```

### 下载结果

**接口**: `GET /download`

**请求格式**:
```http
GET /download?result_dir=results/uuid-string_detect
```

**接口**: `GET /download_original/{file_id}`

**请求格式**:
```http
GET /download_original/uuid-string
```

### API文档

启动服务后，访问 http://localhost:8000/docs 查看完整的交互式API文档。

---

## 🐳 部署方式

### Docker部署（推荐）

#### 1. 使用Docker Compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 2. 使用Docker命令

```bash
# 构建镜像
docker build -t mlsj-app .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/detect:/app/detect \
  -v $(pwd)/segment:/app/segment \
  --name mlsj-app \
  mlsj-app
```

### 本地部署

#### GPU环境

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备模型文件
# 将模型文件放到对应目录

# 3. 启动服务
python main.py
```

#### CPU环境

```bash
# 1. 安装依赖
pip install -r requirements-cpu.txt

# 2. 准备模型文件
# 将模型文件放到对应目录

# 3. 启动服务
python main-cpu.py
```

### 生产环境部署

#### 使用Gunicorn + Uvicorn

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### 使用Nginx反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📝 使用说明

### Web界面使用

1. **上传图片**
   - 点击上传区域或拖拽图片到指定区域
   - 支持JPG、PNG等常见图片格式

2. **选择模型**
   - 在控制面板中选择检测或分割模型
   - 可选模型：YOLO11检测、DETR检测、YOLO11分割、SegNet分割

3. **调节参数**
   - 调整置信度阈值（0.1-0.9）
   - 默认值为0.60

4. **开始预测**
   - 点击"开始预测"按钮
   - 等待模型处理完成

5. **查看结果**
   - 在左右分屏中查看原图和检测结果
   - 查看统计信息和检测详情

6. **下载结果**
   - 点击下载按钮保存结果图片
   - 支持下载检测结果和原始图片
  
<img width="1296" height="943" alt="上传界面" src="./image/上传图片界面.png" />
<img width="1283" height="949" alt="展示界面" src="./image/展示界面.png" />


### API使用示例

#### Python示例

```python
import requests

# 1. 上传图片
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f}
    )
file_id = response.json()['file_id']

# 2. 进行预测
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'file_id': file_id,
        'model_type': 'detect',
        'confidence': 0.60
    }
)
result = response.json()
print(result)
```

#### cURL示例

```bash
# 1. 上传图片
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test_image.jpg"

# 2. 进行预测
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "model_type": "detect",
    "confidence": 0.60
  }'
```

---

## ⚙️ 配置说明

### 模型配置

在 `main.py` 或 `main-cpu.py` 中的 `MODELS` 字典中配置模型：

```python
MODELS = {
    "detect": {
        "name": "YOLO11",
        "path": "detect/yolo11_best.pt",
        "type": "detect"
    },
    "detr": {
        "name": "DETR",
        "path": "detect/detr_recnet50",
        "type": "detect"
    },
    "segment": {
        "name": "YOLO11_seg",
        "path": "segment/yolo11_best.pt",
        "type": "segment"
    },
    "segnet": {
        "name": "segNet",
        "path": "segment/best.pth",
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

### 设备配置

#### GPU模式

- 自动检测CUDA设备
- 使用GPU加速推理
- 需要安装CUDA和cuDNN

#### CPU模式

- 优化线程数和内存使用
- 强制使用CPU：设置环境变量 `CUDA_VISIBLE_DEVICES=-1`
- 使用 `main-cpu.py` 启动

### 环境变量

```bash
# 强制使用CPU
export CUDA_VISIBLE_DEVICES=-1

# 设置端口
export PORT=8000

# 设置主机
export HOST=0.0.0.0
```

---

## 🐛 故障排除

### 常见问题

#### 1. 模型加载失败

**问题**: 模型文件不存在或格式不正确

**解决方案**:
- 检查模型文件是否存在
- 验证模型文件路径是否正确
- 确认模型文件格式（.pt, .pth, .safetensors等）
- 检查PyTorch版本兼容性

#### 2. 内存不足

**问题**: 运行时内存占用过高

**解决方案**:
- 使用CPU版本减少内存占用
- 调整图像输入尺寸
- 关闭不必要的后台进程
- 使用Docker限制内存使用

#### 3. DETR模型错误

**问题**: DETR模型加载或推理失败

**解决方案**:
- 确保transformers库版本兼容（4.57.1）
- 检查模型配置文件完整性
- 确认模型文件目录结构正确
- 检查预处理器配置

#### 4. SegNet权重加载问题

**问题**: SegNet模型权重加载失败

**解决方案**:
- 使用修复版 `test_fixed.py`
- 设置 `weights_only=False` 参数
- 检查模型文件完整性
- 确认模型结构匹配

#### 5. 字体显示问题

**问题**: 中文标签显示为方框

**解决方案**:
- 确保 `MSYH.TTC` 字体文件存在
- 在Linux系统安装中文字体
- 检查字体文件路径

#### 6. 端口占用

**问题**: 端口8000已被占用

**解决方案**:
```bash
# 查找占用端口的进程
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# 修改启动端口
uvicorn main:app --port 8001
```

### 性能优化

#### CPU优化

- 使用 `main-cpu.py` 启动
- 设置 `torch.set_num_threads(1)`
- 限制批次大小
- 使用较小的图像尺寸

#### GPU优化

- 使用混合精度推理
- 启用模型缓存
- 批量处理多个图像
- 使用TensorRT加速（可选）

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# Docker日志
docker-compose logs -f

# 实时查看日志
python start.py
```

---


## 🛠️ 技术栈

### 后端框架

- **FastAPI**: 现代、快速的Web框架
- **Uvicorn**: ASGI服务器
- **Pydantic**: 数据验证

### 深度学习框架

- **PyTorch**: 深度学习框架
- **Ultralytics YOLO11**: 目标检测和分割
- **Transformers**: Hugging Face Transformers库（DETR）
- **OpenCV**: 图像处理
- **Pillow**: 图像处理库

### 前端技术

- **HTML5**: 页面结构
- **CSS3**: 样式设计
- **JavaScript**: 交互逻辑



## 📊 模型性能

| 模型 | 精度 | 速度 | 内存占用 | 应用场景 |
|------|------|------|----------|----------|
| YOLO11检测 | 高 | 快 | 中等 | 实时车辆检测 |
| DETR检测 | 很高 | 中等 | 较高 | 精细部件识别 |
| YOLO11分割 | 高 | 快 | 中等 | 水面区域分割 |
| SegNet分割 | 很高 | 慢 | 较高 | 精确水面分割 |


## 🔄 开发计划

- [ ] 模型训练界面集成
- [ ] 批量处理功能增强
- [ ] 模型性能监控
- [ ] 移动端适配
- [ ] 多语言支持

## 🤝 贡献
ultralytics YOLO
欢迎提交Issue和Pull Request！

## 📄 许可证
MIT License

## 🔬 模型介绍

### YOLO11 模型

- **类型**：目标检测 / 图像分割
- **功能**：车辆检测和水面分割
- **特点**：实时检测，高精度，轻量级
- **应用**：车辆部件识别、水面区域检测
- **优势**：速度快，适合实时应用

### DETR (DEtection TRansformer) 模型

- **类型**：目标检测
- **功能**：车辆部件精细检测
- **特点**：基于Transformer架构，端到端检测
- **应用**：车窗、车门、车轮等车辆部件识别
- **优势**：精度高，无需NMS后处理
- **类别**：车窗、车门、车轮

### SegNet 模型

- **类型**：语义分割
- **功能**：水面语义分割
- **特点**：编码器-解码器架构，像素级分割
- **应用**：积水区域精确分割
- **输出**：水面分割掩码和置信度
- **优势**：分割精度高，边界清晰

---

## 🎯 模型微调指南
### YOLO11
基于Yolo11基础模型进行微调，得到的目标检测模型以及水面分割模型，微调代码切换fine tuning分支，还未合并
微调部分就是YOLO11中的代码，是clone官方开源的方法，修改yaml文件进行自己模型的微调<a href="https://github.com/ultralytics/ultralytics">YOLO官方</a>
下面先展示的是我系统的使用方法与设计，可供大家学习交流。


#### 模型微调
##### 1. 分类依据：
|车窗及以上|车轮顶部至车窗下沿|车轮顶部及以下|
|----|----|----|
|水|其他|水|
<img width="1283" height="949" alt="分类依据" src="./image/分类依据.png" />



##### 2. 数据来源--数据标注
城市洪涝场景下的图像蕴含了积水水面和车辆淹没部位信息，为开展积水自动识别和车辆淹没部位自动判别提供了数据条件。
**标注软件**：X-Anylabeling，第一组目标检测：从图像中标注出矩形边界框和车辆淹没类别，第二组语义分割：从图像中划分出水面多边形边界。
<img width="1283" height="949" alt="标注软件" src="./image/标注软件.png" />

标注完成后，可以创建一个classes.txt文件，里面写三行标注的类别，我们统一用的都是0，1，2，打包导出txt文件。（这里注意软件中标了多少类别，就写多少行类别，多或少都会导出失败）
将标注好的图片分在三个文件中，训练集，验证集，测试集按照6：2：2的比例划分，也是比较经典的分法。
train
   ├── images
   └── labels
val
   ├── images
   └── labels
test
   ├── images
   └── labels

##### 3. 模型训练
一开始想用最小的模型训练出最好的效果，一个是省时间，一个是能够让更多人的各种机器都能训练，具有普遍意义，所以选取的是一个 **yolo11n.pt**。
目标检测模型训练代码，分割模型也类似
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    model.load('目标检测预训练模型/yolo11n.pt')  # 注释则不加载
    results = model.train(
        data='yolo11n_data.yaml',  # 数据集配置文件的路径
        epochs=200,  # 训练轮次总数
        batch=16,  # 批量大小，即单次输入多少图片训练
        imgsz=640,  # 训练图像尺寸
        workers=8,  # 加载数据的工作线程数
        device=0,  # 指定训练的计算设备，无 nvidia 显卡则改为 'cpu'
        optimizer='SGD',  # 训练使用优化器，可选 auto,SGD,Adam,AdamW 等
        amp=True,  # True 或者 False，解释为：自动混合精度(AMP)训练
        cache=False  # True 在内存中缓存数据集图像，服务器推荐开启
    )
```
训练完成后在YOLO11/runs/detect中就能看见训练结果了，训练best.pt存放在其中train文件里取出来用就可以了。
测试模型精度，这边使用的是召回率，精确度，F1分数作为指标
```python
print('mAP50  :', metrics.box.map50)
print('mAP50-95:', metrics.box.map)
print('Precision:', metrics.box.mp)
print('Recall   :', metrics.box.mr)
p, r = metrics.box.mp, metrics.box.mr
f1 = 2 * p * r / (p + r + 1e-16)
print(f'F1        : {f1:.3f}')
```
后面将两个训练好的权重放于系统开发就好了。
| 主要模块     | 版本   |
| ------------ | ------ |
| python       | 3.11.0 |
| CUDA         | 11.8   |
| torch        | 2.7.1  |
| cuDNN        | 90100  |
| numpy        | 1.26.4 |

### DETR模型微调

#### 数据准备
1. **数据格式转换**: 将YOLO格式数据转换为COCO格式
2. **目录结构**:
```bash
DETR/
├── __pycache__/                    # Python编译缓存文件
│   ├── dataset.cpython-310.pyc
│   └── timeT.cpython-310.pyc
├── coco_format_data/               # COCO格式数据集
│   ├── train_annotations.json      # 训练集标注文件
│   ├── val_annotations.json        # 验证集标注文件
│   └── test_annotations.json       # 测试集标注文件
├── detr-r50/                       # DETR-ResNet50模型文件目录
├── dataset.py                      # 数据集加载器（FloodDetectionDataset类）
├── detr_train.py                   # DETR模型训练脚本
├── detr_val.py                     # DETR模型验证脚本
├── download_resnet.py              # ResNet模型下载脚本
├── prediction_result.jpg            # 预测结果示例图片
├── single.py                       # 单张图片预测脚本
├── test_image.jpg                   # 测试图片
├── timeT.py                        # 训练时间跟踪器
└── train.py                        # 训练脚本
```

#### 训练配置
```python
# 主要训练参数
num_epochs = 30
batch_size = 8
learning_rate = 1e-4
image_size = 640
num_classes = 3  # 车窗、车门、车轮
```

#### 训练命令
```bash
cd DETR
python detr_train.py
```

#### 关键特性
- **分层学习率**: backbone使用1e-5，其他层使用1e-4
- **梯度裁剪**: 防止梯度爆炸
- **学习率调度**: 线性学习率调度
- **自动设备检测**: 支持GPU和CPU训练

### SegNet模型微调

#### 数据准备
1. **数据格式**: 需要图像和对应的二值化mask
2. **目录结构**:


### SegNet
```bash
segNet/
├── __pycache__/                    # Python编译缓存文件
│   ├── test.cpython-310.pyc
│   ├── test_fixed.cpython-310.pyc
│   ├── timeT.cpython-310.pyc
│   └── train.cpython-310.pyc
├── runs/                           # 训练运行结果目录
│   └── segnet/
├── segNet_data/                    # SegNet数据集目录
│   ├── train/                      # 训练集
│   │   ├── images/                 # 训练图片
│   │   ├── labels/                 # 标签文件
│   │   └── masks/                  # 掩码图片
│   ├── val/                        # 验证集
│   │   ├── images/                 # 验证图片
│   │   ├── labels/                 # 标签文件
│   │   └── masks/                  # 掩码图片
│   └── test/                       # 测试集
│       ├── images/                 # 测试图片
│       ├── labels/                 # 标签文件
│       └── masks/                  # 掩码图片
├── seg_data.yaml                   # 数据集配置文件
├── test.py                         # 测试脚本（WaterSegmentationPredictor类）
├── test_fixed.py                   # 修复版测试脚本
├── test_image.jpg                   # 测试图片
├── timeT.py                        # 训练时间跟踪器
├── train.py                        # 训练脚本（SegNet模型实现）
├── val.py                          # 验证脚本
└── prediction_result.png            # 预测结果示例图片
```

#### 训练配置
```python
# 主要训练参数
epochs = 200
batch_size = 16
learning_rate = 0.001
img_size = 640
num_classes = 1  # 水面分割
```

#### 训练命令
```bash
cd segNet
python train.py --data-yaml seg_data.yaml --epochs 200 --batch-size 16
```

#### 关键特性
- **数据增强**: 随机翻转、亮度调整
- **IoU评估**: 训练过程中实时计算交并比
- **模型检查点**: 自动保存最佳模型和最新模型
- **学习率调度**: 余弦退火学习率

### 数据集配置

#### DETR数据集配置
```yaml
# 数据集配置文件示例
train_images: "path/to/train/images"
val_images: "path/to/val/images"
categories:
  - id: 0
    name: "车窗"
  - id: 1  
    name: "车门"
  - id: 2
    name: "车轮"
```

#### SegNet数据集配置
```yaml
# seg_data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 1  # 类别数
names: ['水面']  # 类别名称
```

### 训练监控

#### 训练指标
- **DETR**: 训练损失、验证损失、检测精度
- **SegNet**: 训练损失、验证损失、IoU指标

#### 时间跟踪
```python
from timeT import TimeTracker
time_tracker = TimeTracker()
time_tracker.on_train_start()
# 训练过程...
time_tracker.on_train_end()
```

### 模型保存与加载

#### DETR模型保存
```python
model.save_pretrained("detr_vehicle_flood_final")
processor.save_pretrained("detr_vehicle_flood_final")
```

#### SegNet模型保存
```python
torch.save(checkpoint, 'runs/segnet/best.pth')
```

### 性能优化建议

1. **GPU内存优化**:
   - 适当减小batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **训练加速**:
   - 使用多线程数据加载
   - 启用pin_memory加速数据传输
   - 使用预训练权重初始化

3. **模型选择**:
   - DETR适合精细检测任务
   - SegNet适合像素级分割任务
   - 根据任务需求选择合适的模型

// ... existing code ...
