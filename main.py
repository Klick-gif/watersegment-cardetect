import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import json
from ultralytics import YOLO
import uuid
from typing import List, Dict, Any
import shutil
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

app = FastAPI(title="积水识别和车辆淹没部位判别系统")

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# 挂载静态文件
app.mount("/results", StaticFiles(directory="results"), name="results")

# 模板配置
templates = Jinja2Templates(directory="templates")

# 模型配置
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

# 车辆淹没部位类别映射
VEHICLE_PARTS = {
    0: "车窗",
    1: "车门",
    2: "车轮",
}

# 请求模型
class PredictRequest(BaseModel):
    file_id: str
    model_type: str = "detect"
    confidence: float = 0.60

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request, "models": MODELS})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传图片"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{file_id}{file_extension}"
    file_path = os.path.join("uploads", filename)
    
    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"file_id": file_id, "filename": filename, "message": "图片上传成功"}

@app.post("/predict")
async def predict_image(request: PredictRequest):
    """图像预测"""
    try:
        # 从请求中获取参数
        file_id = request.file_id
        model_type = request.model_type
        confidence = request.confidence
        
        print(f"收到预测请求: file_id={file_id}, model_type={model_type}, confidence={confidence}")
        
        # 查找上传的文件
        upload_dir = "uploads"
        files = os.listdir(upload_dir)
        input_file = None
        for f in files:
            if f.startswith(file_id):
                input_file = os.path.join(upload_dir, f)
                break
        
        if not input_file or not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail="找不到上传的文件")
        
        # 加载模型
        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail="不支持的模型类型")
        
        model_path = MODELS[model_type]["path"]
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="模型文件不存在")

        yolo_name = f"{file_id}_{model_type}"
        base_dir = f"results/{file_id}_{model_type}"
        result_dir = base_dir
        index = 1
        while os.path.exists(result_dir):
            yolo_name = f"{yolo_name}_{index}"
            result_dir = f"{base_dir}_{index}"
            index += 1

        os.makedirs(result_dir)
        
        if model_type == "detect" or model_type == "segment":
            model = YOLO(model_path)
            
            # 进行预测
            results = model.predict(
                source=input_file,
                conf=confidence,
                save=True,
                project="results",
                name=yolo_name,
                exist_ok=True
            )
            
            # 处理结果
            result_data = []
            vehicle_stats = {}
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        result_data.append({
                            "class_id": cls,
                            "class_name": VEHICLE_PARTS.get(cls, f"类别{cls}") if model_type == 'detect' else '水面',
                            "confidence": conf,
                            "bbox": xyxy.tolist()
                        })
                        
                        # 统计车辆部位
                        part_name = VEHICLE_PARTS.get(cls, f"类别{cls}") if model_type == 'detect' else '水面'
                        if part_name not in vehicle_stats:
                            vehicle_stats[part_name] = 0
                        vehicle_stats[part_name] += 1
            
        
        elif model_type == "detr":
            # DETR模型预测逻辑
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            processor = DetrImageProcessor.from_pretrained(model_path, local_files_only=True)
            model = DetrForObjectDetection.from_pretrained(model_path, local_files_only=True)
            model.to(device)
            model.eval()
            print(f"成功加载本地DETR模型: {model_path}")
            
            # 加载图像
            image = Image.open(input_file).convert('RGB')
            
            # 进行预测
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
            
            # 后处理
            target_sizes = torch.tensor([image.size[::-1]]).to(device)  # [height, width]
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence
            )[0]
            
            # 解析结果
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels'].cpu().numpy()
            
            # DETR类别名称映射（根据训练数据）
            detr_class_names = ['车窗', '车门', '车轮']  # cc, cm, lt
            
            # 处理结果
            result_data = []
            vehicle_stats = {}
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= confidence:  # 只保留置信度大于阈值的检测结果
                    x_min, y_min, x_max, y_max = box
                    class_name = detr_class_names[label] if label < len(detr_class_names) else f"类别{label}"
                    
                    result_data.append({
                        "class_id": int(label),
                        "class_name": class_name,
                        "confidence": float(score),
                        "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
                    })
                    
                    # 统计车辆部位
                    if class_name not in vehicle_stats:
                        vehicle_stats[class_name] = 0
                    vehicle_stats[class_name] += 1
            
            
            # 在图像上绘制检测结果
            draw = ImageDraw.Draw(image)
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            
            try:
                font = ImageFont.truetype("msyh.ttc", 20)
            except:
                font = ImageFont.load_default()
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score >= confidence:
                    x_min, y_min, x_max, y_max = box
                    class_name = detr_class_names[label] if label < len(detr_class_names) else f"类别{label}"
                    color = colors[label % len(colors)]
                    
                    # 绘制边界框
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                    
                    # 绘制标签
                    label_text = f"{class_name}: {score:.2f}"
                    try:
                        bbox = draw.textbbox((x_min, y_min), label_text, font=font)
                        draw.rectangle(bbox, fill=color)
                        draw.text((x_min, y_min), label_text, fill="white", font=font)
                    except:
                        draw.text((x_min, y_min), label_text, fill=color)
            
            # 保存结果图像
            result_image_path = os.path.join(result_dir, f"{file_id}_{model_type}.jpg")
            image.save(result_image_path)
            print(f"DETR预测结果已保存到: {result_image_path}")
        
        elif model_type == "segnet":
            from segNet.test import WaterSegmentationPredictor
            predictor = WaterSegmentationPredictor(model_path)
            mask, original_image, prob = predictor.predict(input_file, confidence)
            result_image_path = os.path.join(result_dir, f"{file_id}_{model_type}.jpg")
            # 若整体置信度低于阈值，则不输出检测结果
            if float(prob) < float(confidence):
                cv2.imwrite(result_image_path, original_image)
                result_data = []
                vehicle_stats = {"水面": 0}
                print(f"segNet低于阈值(置信度={prob:.2f} < 阈值={confidence:.2f})，已保存原图: {result_image_path}")
            else:
                # 创建彩色mask并叠加
                colored_mask = np.zeros_like(original_image)
                colored_mask[mask > 0] = [0, 255, 0]  # 绿色表示水面
                overlay = original_image.copy()
                overlay[mask > 0] = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)[mask > 0]
                # 在结果图像上绘制类别与置信度
                label_text = f"水面: {float(prob):.2f}"
                overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(overlay_pil)
                font = ImageFont.truetype("msyh.ttc", 26)
                draw.text((10, 30), label_text, font=font, fill=(0, 255, 0))
                overlay = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
                cv2.imwrite(result_image_path, overlay)
                # 处理结果
                result_data = [{
                    "class_id": 0,
                    "class_name": "水面",
                    "confidence": float(prob),
                }]
                # 使用连通组件分析来统计独立的水面区域（跳过背景0）
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                water_areas = []
                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area > 100:
                        water_areas.append(area)
                vehicle_stats = {"水面": len(water_areas)}
                print(f"segNet预测结果已保存到: {result_image_path}")
            
        else:
            print("未开发的模型类型")
        

        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    result_image = f"/{result_dir}/{file}"
                    print(result_image)
                    break
        
        return {
            "success": True,
            "file_id": file_id,
            "model_type": model_type,
            "result_dir": result_dir,
            "detections": result_data,
            "vehicle_stats": vehicle_stats,
            "result_image": result_image,
            "total_detections": len(result_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")



@app.get("/download")
async def download_result(result_dir: str = Query(...)):
    """下载结果图片"""
    if not os.path.exists(result_dir):
        raise HTTPException(status_code=404, detail="结果目录不存在")
    
    # 查找结果图片
    result_files = []
    for file in os.listdir(result_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            result_files.append(file)
    
    if not result_files:
        raise HTTPException(status_code=404, detail="没有找到结果图片")
    
    # 返回第一个结果图片
    result_file = os.path.join(result_dir, result_files[0])
    return FileResponse(
        result_file,
        media_type='application/octet-stream',
        filename=f"result_{os.path.basename(result_file)}"
    )

@app.get("/download_original/{file_id}")
async def download_original(file_id: str):
    """下载原始图片"""
    upload_dir = "uploads"
    files = os.listdir(upload_dir)
    original_file = None
    
    for f in files:
        if f.startswith(file_id):
            original_file = os.path.join(upload_dir, f)
            break
    
    if not original_file or not os.path.exists(original_file):
        raise HTTPException(status_code=404, detail="原始文件不存在")
    
    return FileResponse(
        original_file,
        media_type='application/octet-stream',
        filename=f"original_{file_id}.jpg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
