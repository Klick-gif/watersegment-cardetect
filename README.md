# YOLO11-
基于Yolo11基础模型进行微调，得到的目标检测模型以及水面分割模型，微调代码切换fine tuning分支，还未合并
微调部分就是YOLO11中的代码，是clone官方开源的方法，修改yaml文件进行自己模型的微调<a href="https://github.com/ultralytics/ultralytics">YOLO官方</a>
下面先展示的是我系统的使用方法与设计，可供大家学习交流。

# 一、分析背景：
城市内涝是指由于强降水或连续降水超过城市排水能力致使城市内产生积水灾害的现象。严重的城市洪涝往往会导致大量车辆被水浸泡淹没，造成车辆受损和人员被困。
所以我做了一个基于图像和深度学习的内涝监测。
我们首先做的是一个目标检测场景，传统YOLO的模型只对一些类别进行检测，比如猫、狗、车、人等等，而我们设想的场景是检测那些洪涝时，大雨爆发时车的淹没情况，判断是否安全，或需要拖车避险。
另一个场景就是对水面进行划分，让水面和车、人以及其他的东西划分开，检测水面覆盖的范围。
<img width="1552" height="625" alt="image" src="https://github.com/user-attachments/assets/817df9f5-5581-4879-a3ab-af9f401c7700" />

# 二、模型微调
## 1. 分类依据：
|车窗及以上|车轮顶部至车窗下沿|车轮顶部及以下|
|----|----|----|
|水|其他|水|

<img width="1545" height="606" alt="image" src="https://github.com/user-attachments/assets/187fe31e-a65d-4bdf-b0e0-301eee96b98b" />
## 2. 数据来源--数据标注
城市洪涝场景下的图像蕴含了积水水面和车辆淹没部位信息，为开展积水自动识别和车辆淹没部位自动判别提供了数据条件。
**标注软件**：X-Anylabeling，第一组目标检测：从图像中标注出矩形边界框和车辆淹没类别，第二组语义分割：从图像中划分出水面多边形边界。
<img width="1522" height="489" alt="image" src="https://github.com/user-attachments/assets/87d220e2-5cd6-443c-a4fc-e9a821e613de" />
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

## 3. 模型训练
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

# 三、🌊 积水识别和车辆淹没部位判别系统

## 主要内容
对基础YOLO模型进行微调为特定领域检测模型
基于FastAPI和YOLO11的智能图像分析平台，用于识别积水场景中的车辆淹没部位。
基于图像和深度学习的内涝监测

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
- 车窗
- 车门  
- 车轮

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
- **部署**: Python 3.11

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
ultralytics YOLO
欢迎提交Issue和Pull Request！

## 📞 支持
如有问题，请查看API文档：http://localhost:8000/docs

---
| 主要模块     | 版本     |
| ------------ | ------  |
| python       | 3.11.0  |
| opencv-python| 4.8.1.78|
| Pillow       | 10.1.0  |
| pydantic     | 2.12.3  |
| ultralytics  | 8.3.217 |
| fastapi      | 0.104.1 |



