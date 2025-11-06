# 使用轻量级的Python基础镜像（CPU专用）
FROM python:3.10-slim-bullseye

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
# 禁用CUDA
ENV CUDA_VISIBLE_DEVICES=-1   


# 设置工作目录
WORKDIR /app

# 安装系统依赖（最小化安装）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 复制项目文件
COPY requirements.txt .

# 安装CPU版本的PyTorch和依赖
RUN pip install --upgrade pip
RUN pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 安装其他依赖（排除GPU相关包）
RUN pip install -r requirements.txt --no-deps

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p uploads results static templates

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]